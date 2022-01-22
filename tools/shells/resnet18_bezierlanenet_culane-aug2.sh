#!/bin/bash
# Trained weights: resnet18_bezierlanenet_culane-aug2_20211109.pt
# Training
python main_landet.py --train --config=configs/lane_detection/bezierlanenet/resnet18_culane-aug2.py --mixed-precision
# Predicting lane points for testing
python main_landet.py --test --config=configs/lane_detection/bezierlanenet/resnet18_culane-aug2.py --mixed-precision
# Testing with official scripts
./autotest_culane.sh resnet18_bezierlanenet_culane-aug2 test checkpoints
