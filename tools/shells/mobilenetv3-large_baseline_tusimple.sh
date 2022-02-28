#!/bin/bash
# Trained weights: mobilenetv3-large_baseline_tusimple_20220209.pt
# Training
python main_landet.py --train --config=configs/lane_detection/baseline/mobilenetv3-large_tusimple.py --mixed-precision
# Predicting lane points for testing
python main_landet.py --test --config=configs/lane_detection/baseline/mobilenetv3-large_tusimple.py --mixed-precision
# Testing with official scripts
./autotest_tusimple.sh mobilenetv3-large_baseline_tusimple test checkpoints
