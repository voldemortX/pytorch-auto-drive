#!/bin/bash
# Trained weights: resnet101_baseline_tusimple_20210424.pt
# Training, scale lr by square root on 11G GPU
python main_landet.py --train --config=configs/lane_detection/baseline/resnet101_tusimple.py --mixed-precision
# Predicting lane points for testing
python main_landet.py --test --config=configs/lane_detection/baseline/resnet101_tusimple.py --mixed-precision
# Testing with official scripts
./autotest_tusimple.sh resnet101_baseline_tusimple test checkpoints
