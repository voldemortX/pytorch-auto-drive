#!/bin/bash
# Trained weights: erfnet_baseline_tusimple_20210424.pt
# Training
python main_landet.py --train --config=configs/lane_detection/baseline/erfnet_tusimple.py --mixed-precision
# Predicting lane points for testing
python main_landet.py --test --config=configs/lane_detection/baseline/erfnet_tusimple.py --mixed-precision
# Testing with official scripts
./autotest_tusimple.sh erfnet_baseline_tusimple test checkpoints
