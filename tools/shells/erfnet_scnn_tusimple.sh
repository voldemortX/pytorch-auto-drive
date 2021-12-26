#!/bin/bash
# Trained weights: erfnet_scnn_tusimple_20210202.pt
# Training
python main_landec.py --train --config=configs/lane_detection/scnn/erfnet_tusimple.py --mixed-precision
# Predicting lane points for testing
python main_landec.py --test --config=configs/lane_detection/scnn/erfnet_tusimple.py --mixed-precision
# Testing with official scripts
./autotest_tusimple.sh erfnet_scnn_tusimple test checkpoints
