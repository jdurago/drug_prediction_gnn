# Copyright 2017-2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"). You
# may not use this file except in compliance with the License. A copy of
# the License is located at
#
#     http://aws.amazon.com/apache2.0/
#
# or in the "license" file accompanying this file. This file is
# distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF
# ANY KIND, either express or implied. See the License for the specific
# language governing permissions and limitations under the License.
import ast
import argparse
import logging
import os
import numpy as np
import pandas as pd
import torch

import dgl
from dgl.model_zoo.chem import GCNClassifier
from dgl.data.utils import split_dataset

from torch.utils.data import DataLoader
from torch.nn import BCEWithLogitsLoss
from torch.optim import Adam

from sklearn.metrics import roc_auc_score
from utils import smiles_to_dgl_graph
from tqdm import tqdm

import matplotlib
from sklearn.metrics import roc_curve, roc_auc_score, f1_score, confusion_matrix, precision_recall_curve, f1_score
import seaborn as sns
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

def collate_molgraphs(data):
    """
    Referenced from: https://github.com/dmlc/dgl/blob/master/examples/pytorch/model_zoo/chem/property_prediction/utils.py

    Batching a list of datapoints for dataloader.
    Parameters
    ----------
    data : list of 3-tuples or 4-tuples.
        Each tuple is for a single datapoint, consisting of
        a SMILES, a DGLGraph, all-task labels and optionally
        a binary mask indicating the existence of labels.
    Returns
    -------
    smiles : list
        List of smiles
    bg : BatchedDGLGraph
        Batched DGLGraphs
    labels : Tensor of dtype float32 and shape (B, T)
        Batched datapoint labels. B is len(data) and
        T is the number of total tasks.
    masks : Tensor of dtype float32 and shape (B, T)
        Batched datapoint binary mask, indicating the
        existence of labels. If binary masks are not
        provided, return a tensor with ones.
    """
    assert len(data[0]) in [3, 4], \
        'Expect the tuple to be of length 3 or 4, got {:d}'.format(len(data[0]))
    if len(data[0]) == 3:
        smiles, graphs, labels = map(list, zip(*data))
        masks = None
    else:
        smiles, graphs, labels, masks = map(list, zip(*data))

    bg = dgl.batch(graphs)
    bg.set_n_initializer(dgl.init.zero_initializer)
    bg.set_e_initializer(dgl.init.zero_initializer)
    labels = torch.stack(labels, dim=0)

    if masks is None:
        masks = torch.ones(labels.shape)
    else:
        masks = torch.stack(masks, dim=0)
    return smiles, bg, labels, masks


class Meter(object):
    """
    Referenced from: https://github.com/dmlc/dgl/blob/master/examples/pytorch/model_zoo/chem/property_prediction/utils.py

    Track and summarize model performance on a dataset for
    (multi-label) binary classification."""

    def __init__(self):
        self.mask = []
        self.y_pred = []
        self.y_true = []

    def update(self, y_pred, y_true, mask):
        """Update for the result of an iteration
        Parameters
        ----------
        y_pred : float32 tensor
            Predicted molecule labels with shape (B, T),
            B for batch size and T for the number of tasks
        y_true : float32 tensor
            Ground truth molecule labels with shape (B, T)
        mask : float32 tensor
            Mask for indicating the existence of ground
            truth labels with shape (B, T)
        """
        self.y_pred.append(y_pred.detach().cpu())
        self.y_true.append(y_true.detach().cpu())
        self.mask.append(mask.detach().cpu())

    def roc_auc_score(self):
        """Compute roc-auc score for each task.
        Returns
        -------
        list of float
            roc-auc score for all tasks
        """
        mask = torch.cat(self.mask, dim=0)
        y_pred = torch.cat(self.y_pred, dim=0)
        y_true = torch.cat(self.y_true, dim=0)
        # Todo: support categorical classes
        # This assumes binary case only
        y_pred = torch.sigmoid(y_pred)
        n_tasks = y_true.shape[1]
        scores = []
        for task in range(n_tasks):
            task_w = mask[:, task]
            task_y_true = y_true[:, task][task_w != 0].numpy()
            task_y_pred = y_pred[:, task][task_w != 0].numpy()
            scores.append(roc_auc_score(task_y_true, task_y_pred))
        return scores

    def l1_loss(self, reduction):
        """Compute l1 loss for each task.
        Returns
        -------
        list of float
            l1 loss for all tasks
        reduction : str
            * 'mean': average the metric over all labeled data points for each task
            * 'sum': sum the metric over all labeled data points for each task
        """
        mask = torch.cat(self.mask, dim=0)
        y_pred = torch.cat(self.y_pred, dim=0)
        y_true = torch.cat(self.y_true, dim=0)
        n_tasks = y_true.shape[1]
        scores = []
        for task in range(n_tasks):
            task_w = mask[:, task]
            task_y_true = y_true[:, task][task_w != 0]
            task_y_pred = y_pred[:, task][task_w != 0]
            scores.append(F.l1_loss(task_y_true, task_y_pred, reduction=reduction).item())
        return scores

    def rmse(self):
        """Compute RMSE for each task.
        Returns
        -------
        list of float
            rmse for all tasks
        """
        mask = torch.cat(self.mask, dim=0)
        y_pred = torch.cat(self.y_pred, dim=0)
        y_true = torch.cat(self.y_true, dim=0)
        n_data, n_tasks = y_true.shape
        scores = []
        for task in range(n_tasks):
            task_w = mask[:, task]
            task_y_true = y_true[:, task][task_w != 0]
            task_y_pred = y_pred[:, task][task_w != 0]
            scores.append(np.sqrt(F.mse_loss(task_y_pred, task_y_true).cpu().item()))
        return scores

    def compute_metric(self, metric_name, reduction='mean'):
        """Compute metric for each task.
        Parameters
        ----------
        metric_name : str
            Name for the metric to compute.
        reduction : str
            Only comes into effect when the metric_name is l1_loss.
            * 'mean': average the metric over all labeled data points for each task
            * 'sum': sum the metric over all labeled data points for each task
        Returns
        -------
        list of float
            Metric value for each task
        """
        assert metric_name in ['roc_auc', 'l1', 'rmse'], \
            'Expect metric name to be "roc_auc", "l1" or "rmse", got {}'.format(metric_name)
        assert reduction in ['mean', 'sum']
        if metric_name == 'roc_auc':
            return self.roc_auc_score()
        if metric_name == 'l1':
            return self.l1_loss(reduction)
        if metric_name == 'rmse':
            return self.rmse()

def generate_confusion_matrix_plot(y_true: np.array, logits: torch.Tensor) -> (matplotlib.figure.Figure, np.array):
    fig = plt.figure()
    ax= plt.subplot()
    y_pred = torch.clamp(logits, min=0.0, max=1.0).round().detach().numpy()
    cm = confusion_matrix(y_true, y_pred)
    hm = sns.heatmap(cm, annot=True, ax = ax, fmt='g')
    # labels, title and ticks
    ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels'); 
    ax.set_title('Confusion Matrix'); 
    
    return fig, cm

def generate_auc_roc_plot(y_true: np.array, logits: torch.Tensor) -> (matplotlib.figure.Figure, np.array):
    ## Generate ROC-AUC Curve based on validation set
    fig, ax = plt.subplots(1)
    tpr, fpr, _ = roc_curve(y_true, logits.detach().numpy())
    auc = roc_auc_score(y_true, logits.detach().numpy())
    plt.plot(tpr, fpr)
    ax.get_xaxis().set_visible(True)
    ax.get_yaxis().set_visible(True)
    ax.text(0.95, 0.01, u"%0.2f" % auc,
            verticalalignment='bottom', horizontalalignment='right',
            transform=ax.transAxes, weight='bold',
            fontsize=10)
    plt.suptitle('Validation AUC-ROC Curve')
    
    return fig, auc

def generate_precision_recall_plot(y_true: np.array, logits: torch.Tensor) -> (matplotlib.figure.Figure, np.array):
    fig, ax = plt.subplots(1)
    proba = logits.detach().numpy()
    pred = torch.clamp(logits, min=0.0, max=1.0).round().detach().numpy()
    precision, recall, _ = precision_recall_curve(y_true, proba)
    f1_val = f1_score(y_true, pred)
    plt.plot(recall, precision)
    ax.get_xaxis().set_visible(True)
    ax.get_yaxis().set_visible(True)
    ax.text(0.95, 0.01, u"%0.2f" % f1_val,
            verticalalignment='bottom', horizontalalignment='right',
            transform=ax.transAxes, weight='bold',
            fontsize=10)
    plt.suptitle('Validation Precision-Recall Curve')
    
    return fig, f1_val

def main(args):
    # Download Data
    if args.dev_mode.lower() == 'true':
        df = pd.read_csv(os.path.join(args.data_dir, 'dev_HIV.csv'))
    else:
        df = pd.read_csv(os.path.join(args.data_dir, 'HIV.csv')) 
    print(f'Num data points: {len(df)}')
    y = df['HIV_active'].tolist() # get labels for stratified splitting of train/test/split sets

    # Format Data for Ingestion Into Pytorch Dataloader
    data = df.to_records(index=False)  # creates tuple with format (smiles, activity, label)
    data = [(s, a, l, smiles_to_dgl_graph(s)) for s, a, l in tqdm(data, total=len(data))]  # tuple with format (smiles, activity, label, dgl.graph)
    data = [(s, g, torch.tensor([l], dtype=torch.float)) for s, a, l, g in tqdm(data, total=len(data))]  # add feature tensor to dataset

    # Create Model
    in_feats = args.in_feats
    gcn_hidden_feats = [args.gcn_hidden_feats] * args.num_hidden_layers # number of hidden layers and features for each layer. Format is -> list[int, int, ...]
    classifier_hidden_feats = args.classifier_hidden_feats
    n_tasks = args.n_tasks # n_tasks is the number of output features (12 for Tox21, 1 for HIV dataset)
    batch_size = args.batch_size
    atom_data_field = args.atom_data_field
    loss_criterion = BCEWithLogitsLoss()
    learning_rate = args.learning_rate 
    metric_name = args.metric_name
    epochs = args.epochs
    
    model = GCNClassifier(in_feats=in_feats,
                      gcn_hidden_feats=gcn_hidden_feats,
                      classifier_hidden_feats=classifier_hidden_feats,
                      n_tasks=n_tasks)

    # Generate train/test/val data sets
    train, test, y_train, y_test = train_test_split(data,y, shuffle=True, stratify=y, test_size=0.1, random_state=args.random_state) # split data into train and test
    train, val, y_train, y_val = train_test_split(data,y, shuffle=True, stratify=y, test_size=0.11) # further split train into train validation
    train_loader = DataLoader(train, batch_size=batch_size, collate_fn=collate_molgraphs)
    test_loader = DataLoader(test, batch_size=batch_size, collate_fn=collate_molgraphs)
    val_loader = DataLoader(val, batch_size=len(val), collate_fn=collate_molgraphs)

    # Train The Model
    for e in range(epochs):
        # Train the model on batch of data

        for batch_id, batch_data in enumerate(train_loader):
            smiles, bg, labels, masks = batch_data
            atom_feats = bg.ndata.pop(atom_data_field)
            logits = model(bg, atom_feats)
            loss = (loss_criterion(logits, labels) * (masks != 0).float()).mean()
            optimizer = Adam(model.parameters(), lr=learning_rate)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
        # Eval the model on test set
        eval_meter = Meter()
        with torch.no_grad():
            for batch_id, batch_data in enumerate(test_loader):
                smiles, bg, labels, masks = batch_data
                atom_feats = bg.ndata.pop(atom_data_field)
    #             atom_feats, labels = atom_feats.to(args['device']), labels.to(args['device'])
                logits = model(bg, atom_feats)
                eval_meter.update(logits, labels, masks)
        
        test_score = np.mean(eval_meter.compute_metric(metric_name))
        print(f'epoch:{e}, train_loss:{loss:.4f}, test_score:{test_score}')

    print('Done Training!')
    
    # Save the model
    torch.save(model, os.path.join(args.model_dir, 'gnn_model.pt'))

    # Score the model based on validation data
    val_meter = Meter()
    val_loader = DataLoader(val, batch_size=len(val), collate_fn=collate_molgraphs)
    
    for i, batch_data in enumerate(val_loader):
        smiles, bg, labels, masks = batch_data
        atom_feats = bg.ndata.pop(atom_data_field)
        logits = model(bg, atom_feats)

    cm_fig, _ = generate_confusion_matrix_plot(labels, logits)
    cm_fig.savefig(os.path.join(args.model_dir, 'confusion_matrix.png'))

    auc_roc_fig, auc_roc = generate_auc_roc_plot(labels, logits)
    auc_roc_fig.savefig(os.path.join(args.model_dir, 'auc_roc_validation.png') )

    rp_curve_fig, f1_score = generate_precision_recall_plot(labels, logits)
    rp_curve_fig.savefig(os.path.join(args.model_dir, 'recall_precision_validation.png'))

    print(f'auc_roc:{auc_roc}, f1_score:{f1_score}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dev-mode', type=str, default='False')
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--learning-rate', type=float, default=1e-4)
    parser.add_argument('--in-feats', type=int, default=74, help='num features per node (default: 74 for chemistry data)')
    parser.add_argument('--gcn-hidden-feats', type=int, default=64)
    parser.add_argument('--classifier-hidden-feats', type=int, default=64)
    parser.add_argument('--n-tasks', type=int, default=1, help='output size of classification layer')
    parser.add_argument('--batch-size', type=int, default=200)
    parser.add_argument('--atom-data-field', type=str, default='h')
    parser.add_argument('--metric-name', type=str, default='roc_auc')
    parser.add_argument('--num-hidden-layers', type=int, default=2) 
    parser.add_argument('--random-state', type=int, default=-1, help='random state for train/test/split dataset. If -1 then defaults to RandomState') 
    
    # The parameters below retrieve their default values from SageMaker environment variables, which are
    # instantiated by the SageMaker containers framework.
    # https://github.com/aws/sagemaker-containers#how-a-script-is-executed-inside-the-container
    parser.add_argument('--hosts', type=str, default=ast.literal_eval(os.environ['SM_HOSTS']))
    parser.add_argument('--current-host', type=str, default=os.environ['SM_CURRENT_HOST'])
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--data-dir', type=str, default=os.environ['SM_CHANNEL_TRAINING'])
    parser.add_argument('--num-gpus', type=int, default=os.environ['SM_NUM_GPUS'])

    args = parser.parse_args()

    if args.random_state == -1: args.random_state = None
    main(args)
