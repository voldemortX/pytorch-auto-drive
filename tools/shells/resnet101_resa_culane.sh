#!/bin/bash
# Trained weights: resnet101_resa_culane_20211016.pt
# Training
python -m torch.distributed.launch --nproc_per_node=8 --use_env main_landet.py --train --config=configs/lane_detection/resa/resnet101_culane.py
# Predicting lane points for testing
python main_landet.py --test --config=configs/lane_detection/resa/resnet101_culane.py
# Testing with official scripts
./autotest_culane.sh resnet101_resa_culane test checkpoints
