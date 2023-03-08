#!/bin/bash

git clone https://github.com/semantic-systems/coypu-current-events-for-ml.git
git clone https://github.com/semantic-systems/current-events-to-kg.git
ln -s ../../../current-events-to-kg/currenteventstokg coypu-current-events-for-ml/ce4ml/datasets/currenteventstokg

cd current-events-for-ml

pip install -r requirements.txt
pip install tensorflow
pip install torch --extra-index-url https://download.pytorch.org/whl/cu116 --force-reinstall --ignore-installed

# CUDA_VISIBLE_DEVICES=0 python -m ce4ml.ml --warmup_length 0.1