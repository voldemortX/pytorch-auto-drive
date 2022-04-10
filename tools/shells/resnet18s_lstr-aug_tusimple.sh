#!/bin/bash
# Trained weights: resnet18s_lstr-aug_tusimple_20210629.pt
# Training
python main_landet.py --train --config=configs/lane_detection/lstr/resnet18s_tusimple_aug.py
# Predicting lane points for testing
python main_landet.py --test --config=configs/lane_detection/lstr/resnet18s_tusimple_aug.py
# Testing with official scripts
./autotest_tusimple.sh resnet18s_lstr-aug_tusimple test checkpoints
