#!/bin/bash
# Trained weights: erfnet_baseline_culane_20210204.pt
# Training
python main_landet.py --train --config=configs/lane_detection/baseline/erfnet_culane.py --mixed-precision
# Predicting lane points for testing
python main_landet.py --test --config=configs/lane_detection/baseline/erfnet_culane.py --mixed-precision
# Testing with official scripts
./autotest_culane.sh erfnet_baseline_culane test checkpoints
