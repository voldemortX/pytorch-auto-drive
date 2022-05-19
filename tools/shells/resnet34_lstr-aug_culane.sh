#!/bin/bash
# Trained weights: resnet34_lstr-aug_culane_20211012.pt
# Training
python main_landet.py --train --config=configs/lane_detection/lstr/resnet34_culane_aug.py
# Predicting lane points for testing
python main_landet.py --test --config=configs/lane_detection/lstr/resnet34_culane_aug.py
# Testing with official scripts
./autotest_culane.sh resnet34_lstr_culane-aug test checkpoints
