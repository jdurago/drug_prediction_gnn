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

# For more information on creating a Dockerfile
# https://docs.docker.com/compose/gettingstarted/#step-2-create-a-dockerfile
# https://github.com/awslabs/amazon-sagemaker-examples/master/advanced_functionality/pytorch_extending_our_containers/pytorch_extending_our_containers.ipynb
ARG REGION=us-west-2

# FROM continuumio/miniconda3
# FROM nvidia/cuda
FROM nvidia/cuda:10.1-base-ubuntu16.04
RUN apt-get update
RUN apt-get -y install curl
# Install miniconda to /miniconda
RUN curl -LO http://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh
RUN bash Miniconda3-latest-Linux-x86_64.sh -p /miniconda -b
RUN rm Miniconda3-latest-Linux-x86_64.sh
ENV PATH=/miniconda/bin:${PATH}
RUN conda update -y conda

# Try using this image
# FROM 763104351884.dkr.ecr.$REGION.amazonaws.com/sagemaker-pytorch:0.4.0-cpu-py3
#FROM pytorch/pytorch:latest

RUN apt-get update

# install packages including GCC
RUN apt-get install -y build-essential 
RUN conda install -y -c conda-forge rdkit==2018.09.3
RUN conda install pytorch torchvision -y -c pytorch
RUN pip install sagemaker-containers
RUN conda install -y -c conda-forge matplotlib
#RUN pip install git+https://github.com/samoturk/mol2vec
RUN pip install scikit-learn==0.21.3 dgl-cu101==0.4.1 seaborn

ENV PATH="/opt/ml/code:${PATH}"

# /opt/ml and all subdirectories are utilized by SageMaker, we use the /code subdirectory to store our user code.
COPY /src /opt/ml/code

# this environment variable is used by the SageMaker PyTorch container to determine our user code directory.
ENV SAGEMAKER_SUBMIT_DIRECTORY /opt/ml/code

# this environment variable is used by the SageMaker PyTorch container to determine our program entry point
# for training and serving.
# For more information: https://github.com/aws/sagemaker-pytorch-container

ENV SAGEMAKER_PROGRAM model.py
