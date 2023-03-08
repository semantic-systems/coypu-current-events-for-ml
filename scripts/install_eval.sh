#!/bin/bash

# anaconda prep
conda create -n blink37 -y python=3.7
conda activate blink37

# current-events-for-ml
git clone https://github.com/semantic-systems/coypu-current-events-for-ml.git
cd coypu-current-events-for-ml

# blink/elq
git clone https://github.com/facebookresearch/BLINK.git
ln -s ./BLINK/blink/ blink
ln -s ./BLINK/elq/ elq
ln -s ../../BLINK/ ce4ml/eval/BLINK
./BLINK/download_blink_models.sh
./BLINK/download_elq_models.sh
mkdir models # needed by elq

# install modules
pip install -r requirements.txt
pip install -r BLINK/requirements.txt
pip install torch --extra-index-url https://download.pytorch.org/whl/cu116 --force-reinstall --ignore-installed


# CUDA_VISIBLE_DEVICES=0 python -m ce4ml.eval -m blink
# CUDA_VISIBLE_DEVICES=0 python -m ce4ml.eval -m elq








