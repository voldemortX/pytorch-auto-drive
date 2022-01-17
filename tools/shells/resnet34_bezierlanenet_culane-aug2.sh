#!/bin/bash
# Trained weights: resnet34_bezierlanenet_culane_2021xxxx.pt
# Training
python main_landet.py --train --config=configs/lane_detection/bezierlanenet/resnet34_culane-aug2.py --mixed-precision
# Predicting lane points for testing
python main_landet.py --test --config=configs/lane_detection/bezierlanenet/resnet34_culane-aug2.py --mixed-precision
# Testing with official scripts
./autotest_culane.sh resnet34_bezierlanenet_culane-aug2 test checkpoints
