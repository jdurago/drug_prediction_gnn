#!/usr/bin/env bash

# This script downloads training data and saves locally for local training

mkdir data

# download pretrained mol2vec model
wget -O data/model_300dim.pkl https://github.com/samoturk/mol2vec/raw/master/examples/models/model_300dim.pkl model_300dim.pkl   
# download example data     
wget -O data/ames.sdf https://github.com/samoturk/mol2vec/raw/master/examples/data/ames.sdf
# download hiv data
wget -O data/hiv.zip https://s3-us-west-1.amazonaws.com/deepchem.io/datasets/molnet_publish/hiv.zip
cd data; unzip hiv.zip
