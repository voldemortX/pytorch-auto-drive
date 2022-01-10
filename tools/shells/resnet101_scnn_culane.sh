#!/bin/bash
# Trained weights: resnet101_scnn_culane_20210314.pt
# Training, scale lr linearly on 11G GPU (square root scaling does not converge on this dataset)
python main_landet.py --train --config=configs/lane_detection/scnn/resnet101_culane.py --mixed-precision
# Predicting lane points for testing
python main_landet.py --test --config=configs/lane_detection/scnn/resnet101_culane.py --mixed-precision
# Testing with official scripts
./autotest_culane.sh resnet101_scnn_culane test checkpoints
