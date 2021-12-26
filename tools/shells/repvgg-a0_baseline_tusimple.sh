#!/bin/bash
# Trained weights: repvgg-a0_baseline_tusimple_20211226.pt
# Training
python main_landec.py --train --mixed-precision --config configs/lane_detection/baseline/repvgg-a0_tusimple.py
# Predicting lane points for testing
python main_landec.py --train --mixed-precision --config configs/lane_detection/baseline/repvgg-a0_tusimple.py
# Testing with official scripts
./autotest_tusimple.sh repvgg-a0_baseline_tusimple test