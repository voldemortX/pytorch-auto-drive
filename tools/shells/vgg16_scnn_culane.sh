#!/bin/bash
# Trained weights: vgg16_scnn_culane_20210309.pt
# Training
python main_landet.py --train --config=configs/lane_detection/scnn/vgg16_culane.py --mixed-precision
# Predicting lane points for testing
python main_landet.py --test --config=configs/lane_detection/scnn/vgg16_culane.py --mixed-precision
# Testing with official scripts
./autotest_culane.sh vgg16_scnn_culane test checkpoints
