#!/bin/bash
# Trained weights: enet_baseline_tusimple_20210312.pt
python main_landet.py --train --config=configs/lane_detection/baseline/enet_tusimple.py --mixed-precision
# Predicting lane points for testing
python main_landet.py --test --config=configs/lane_detection/baseline/enet_tusimple.py --mixed-precision
# Testing with official scripts
./autotest_tusimple.sh enet_baseline_tusimple test checkpoints
