#!/bin/bash
# Trained weights: resnet34_resa_tusimple_20211019.pt
# Training
python -m torch.distributed.launch --nproc_per_node=4 --use_env main_landec.py --train --config=configs/lane_detection/resa/resnet34_tusimple.py
# Predicting lane points for testing
python main_landec.py --test --config=configs/lane_detection/resa/resnet34_tusimple.py
# Testing with official scripts
./autotest_tusimple.sh resnet34_resa_tusimple test checkpoints
