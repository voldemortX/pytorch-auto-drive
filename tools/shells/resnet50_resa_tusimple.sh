#!/bin/bash
# Trained weights: resnet50_resa_tusimple_20211019.pt
# Training
python -m torch.distributed.launch --nproc_per_node=4 --use_env main_landet.py --train --config=configs/lane_detection/resa/resnet50_tusimple.py
# Predicting lane points for testing
python main_landet.py --test --config=configs/lane_detection/resa/resnet50_tusimple.py
# Testing with official scripts
./autotest_tusimple.sh resnet50_resa_tusimple test checkpoints
