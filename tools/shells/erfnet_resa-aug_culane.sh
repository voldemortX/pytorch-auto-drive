#!/bin/bash
# TODO try multi-GPU training
# Training
python -m torch.distributed.launch --nproc_per_node=2 --use_env main_landet.py --train --config=configs/lane_detection/resa/erfnet_culane.py 
# Predicting lane points for testing
python main_landet.py --test --config=configs/lane_detection/resa/erfnet_culane.py 
# Testing with official scripts
./autotest_culane.sh erfnet_resa_culane test checkpoints
