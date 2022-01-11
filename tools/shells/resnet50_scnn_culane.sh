#!/bin/bash
# Trained weights: resnet50_scnn_culane_20210311.pt
# Training, scale lr linearly on 11G GPU (square root scaling does not converge on this dataset)
python main_landet.py --train --config=configs/lane_detection/scnn/resnet50_culane.py --mixed-precision
# Predicting lane points for testing
python main_landet.py --test --config=configs/lane_detection/scnn/resnet50_culane.py --mixed-precision
# Testing with official scripts
./autotest_culane.sh resnet50_scnn_culane test checkpoints
