#!/bin/bash
# Trained weights: resnet18s_lstr_tusimple_20210701.pt
# Training
python main_landet.py --train --config=configs/lane_detection/lstr/resnet18s_tusimple.py
# Predicting lane points for testing
python main_landet.py --test --config=configs/lane_detection/lstr/resnet18s_tusimple.py
# Testing with official scripts
./autotest_tusimple.sh resnet18s_lstr_tusimple test checkpoints
