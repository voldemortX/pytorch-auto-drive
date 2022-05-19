#!/bin/bash
# Training
python main_landet.py --train --config=configs/lane_detection/resa/erfnet_tusimple.py 
# Predicting lane points for testing
python main_landet.py --test --config=configs/lane_detection/resa/erfnet_tusimple.py 
# Testing with official scripts
./autotest_tusimple.sh erfnet_resa_tusimple test checkpoints
