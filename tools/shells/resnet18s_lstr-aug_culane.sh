#!/bin/bash
# Trained weights: resnet18s_lstr-aug_culane_20210721.pt
# Training
python main_landet.py --train --config=configs/lane_detection/lstr/resnet18s_culane_aug.py
# Predicting lane points for testing
python main_landet.py --test --config=configs/lane_detection/lstr/resnet18s_culane_aug.py
# Testing with official scripts
./autotest_culane.sh resnet18s_lstr-aug_culane test checkpoints
