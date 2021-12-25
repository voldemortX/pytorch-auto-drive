#!/bin/bash
# Trained weights: resnet18_resa_culane_20211016.pt
# Training
python -m torch.distributed.launch --nproc_per_node=4 --use_env main_landec.py --train --config=configs/lane_detection/resa/resnet18_culane.py
# Predicting lane points for testing
python main_landec.py --test --config=configs/lane_detection/resa/resnet18_culane.py
# Testing with official scripts
./autotest_culane.sh resnet18_resa_culane test checkpoints
