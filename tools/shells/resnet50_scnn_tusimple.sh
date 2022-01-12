#!/bin/bash
# Trained weights: resnet50_scnn_tusimple_20210217.pt
# Training, scale lr by square root on 11G GPU
python main_landet.py --train --config=configs/lane_detection/scnn/resnet50_tusimple.py --mixed-precision
# Predicting lane points for testing
python main_landet.py --test --config=configs/lane_detection/scnn/resnet50_tusimple.py --mixed-precision
# Testing with official scripts
./autotest_tusimple.sh resnet50_scnn_tusimple test checkpoints
