#!/bin/bash
# Trained weights: resnet34_scnn_culane_20210220.pt
# Training
python main_landet.py --train --config=configs/lane_detection/scnn/resnet34_culane.py --mixed-precision
# Predicting lane points for testing
python main_landet.py --test --config=configs/lane_detection/scnn/resnet34_culane.py --mixed-precision
# Testing with official scripts
./autotest_culane.sh resnet34_scnn_culane test checkpoints
