#!/bin/bash
# Trained weights: mobilenetv2_baseline_culane_20220209.pt
# Training
python main_landet.py --train --config=configs/lane_detection/baseline/mobilenetv2_culane.py --mixed-precision
# Predicting lane points for testing
python main_landet.py --test --config=configs/lane_detection/baseline/mobilenetv2_culane.py --mixed-precision
# Testing with official scripts
./autotest_culane.sh mobilenetv2_baseline_culane test checkpoints
