#!/bin/bash
# Trained weights: enet_baseline_culane_20210312.pt
# Training
python main_landec.py --train --config=configs/lane_detection/baseline/enet_culane.py --mixed-precision
# Predicting lane points for testing
python main_landec.py --test --config=configs/lane_detection/baseline/enet_culane.py --mixed-precision
# Testing with official scripts
./autotest_culane.sh enet_baseline_culane test checkpoints
