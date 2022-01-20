#!/bin/bash
# Trained weights: resnet34_bezierlanenet_tusimple_2021xxxx.pt
# Training
python main_landet.py --train --config=configs/lane_detection/bezierlanenet/resnet34_tusimple-aug2.py
# Predicting lane points for testing
python main_landet.py --test --config=configs/lane_detection/bezierlanenet/resnet34_tusimple-aug2.py
# Testing with official scripts
./autotest_tusimple.sh resnet34_bezierlanenet_tusimple-aug2 test checkpoints
