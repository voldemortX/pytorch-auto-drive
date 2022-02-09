#!/bin/bash
# Trained weights: mobilenetv3-large_resa_culane_20220209.pt
# Training
python main_landet.py --train --config=configs/lane_detection/resa/mobilenetv3-large_culane.py --mixed-precision
# Predicting lane points for testing
python main_landet.py --test --config=configs/lane_detection/resa/mobilenetv3-large_culane.py --mixed-precision
# Testing with official scripts
./autotest_culane.sh mobilenetv3-large_resa_culane test checkpoints
