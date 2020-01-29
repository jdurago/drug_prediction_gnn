import pytest
import numpy as np
import torch
from model import generate_confusion_matrix_plot, generate_auc_roc_plot, generate_precision_recall_plot 

def test_generate_confusion_matrix_plot():
    labels = [1, 0, 1, 1, 0, 1]
    logits = torch.tensor([-0.2, -0.2, 1.2, 1.2, -0.01, 0.9])
    cm_fig, cm = generate_confusion_matrix_plot(labels, logits)
    expected_array = np.array([[2, 0],[1, 3]])
    assert  np.array_equal(cm,expected_array )

def test_generate_auc_roc_plot():
    y_true = np.array([0, 0, 1, 1])
    logits = torch.tensor([0.1, 0.4, 0.35, 0.8])
    
    auc_roc_fig, auc_roc = generate_auc_roc_plot(y_true, logits) 
    assert auc_roc == 0.75

def test_generate_precision_recall_plot():
    labels = [0, 1, 1, 0, 1, 1]
    logits = torch.tensor([-0.1, 0.9, 1.1, 0.01, 0.02, 1.0])
    rp_curve_fig, f1_score = generate_precision_recall_plot(labels, logits)
    assert f1_score == 0.888888888888889


    labels = [0, 0, 0, 0, 0, 1]
    logits = torch.tensor([-0.1, -0.01, -0.2, -0.3,-0.02, 1.0])
    rp_curve_fig, f1_score = generate_precision_recall_plot(labels, logits)
    assert f1_score == 1.0 

