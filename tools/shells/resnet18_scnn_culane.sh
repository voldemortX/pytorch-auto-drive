#!/bin/bash
# Trained weights: resnet18_scnn_culane_20210222.pt
# Training
python main_landet.py --train --config=configs/lane_detection/scnn/resnet18_culane.py --mixed-precision
# Predicting lane points for testing
python main_landet.py --test --config=configs/lane_detection/scnn/resnet18_culane.py --mixed-precision
# Testing with official scripts
./autotest_culane.sh resnet18_scnn_culane test checkpoints
