#!/bin/bash
# Trained weights: mobilenetv2_resa_tusimple_20220209.pt
# Training
python main_landet.py --train --config=configs/lane_detection/resa/mobilenetv2_tusimple.py --mixed-precision
# Predicting lane points for testing
python main_landet.py --test --config=configs/lane_detection/resa/mobilenetv2_tusimple.py --mixed-precision
# Testing with official scripts
./autotest_tusimple.sh mobilenetv2_resa_tusimple test checkpoints
