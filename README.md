# Drug Prediction
Implementation of graph neural networks for the [HIV MoleculeNet dataset](http://moleculenet.ai/datasets-1). This package is meant to be run as an AWS Sagemaker training job.

# Build Model and Train
Sagemaker training jobs utilize docker containers stored in AWS Elastic Container Registry to execute. Code for running a local and container based training jobs can be found in the <b> [run_container.ipynb](./run_container.ipynb) </b>

Steps for running a training job are as follows:
1. Build container and upload to AWS ECR using the build_and_push.sh script <br />
    `!cd container; ./build_and_push.sh drug-prediction-gnn`
2. Download datasets by using get_data.sh: <br />
    `./get_data.sh`
3. Refer to [run_container.ipynb](./run_container.ipynb) on running local training job and a Sagemaker training job

