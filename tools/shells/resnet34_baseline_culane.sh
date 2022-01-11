#!/bin/bash
# Trained weights: resnet34_baseline_culane_20210219.pt
# Training
python main_landet.py --train --config=configs/lane_detection/baseline/resnet34_culane.py --mixed-precision
# Predicting lane points for testing
python main_landet.py --test --config=configs/lane_detection/baseline/resnet34_culane.py --mixed-precision
# Testing with official scripts
./autotest_culane.sh resnet34_baseline_culane test checkpoints
