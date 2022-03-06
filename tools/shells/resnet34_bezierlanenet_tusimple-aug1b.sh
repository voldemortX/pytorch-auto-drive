#!/bin/bash
# Trained weights: resnet34_bezierlanenet_tusimple-aug1b_20211109.pt
# Training
python main_landet.py --train --config=configs/lane_detection/bezierlanenet/resnet34_tusimple-aug1b.py
# Predicting lane points for testing
python main_landet.py --test --config=configs/lane_detection/bezierlanenet/resnet34_tusimple-aug1b.py
# Testing with official scripts
./autotest_tusimple.sh resnet34_bezierlanenet_tusimple-aug1b test checkpoints
