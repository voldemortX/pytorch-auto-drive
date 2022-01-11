#!/bin/bash
# Trained weights: resnet34_scnn_tusimple_20210424.pt
# Training
python main_landet.py --train --config=configs/lane_detection/scnn/resnet34_tusimple.py --mixed-precision
# Predicting lane points for testing
python main_landet.py --test --config=configs/lane_detection/scnn/resnet34_tusimple.py --mixed-precision
# Testing with official scripts
./autotest_tusimple.sh resnet34_scnn_tusimple test checkpoints
