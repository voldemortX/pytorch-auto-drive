#!/bin/bash
# Trained weights: erfnet_scnn_culane_20210206.pt
# Training
python main_landet.py --train --config=configs/lane_detection/scnn/erfnet_culane.py --mixed-precision
# Predicting lane points for testing
python main_landet.py --test --config=configs/lane_detection/scnn/erfnet_culane.py --mixed-precision
# Testing with official scripts
./autotest_culane.sh erfnet_scnn_culane test checkpoints
