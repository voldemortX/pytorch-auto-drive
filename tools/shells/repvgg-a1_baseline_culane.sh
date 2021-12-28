#!/bin/bash
# Trained weights: repvgg-a1_baseline_culane_20211228.pt
# Training
python main_landec.py --train --mixed-precision --config configs/lane_detection/baseline/repvgg-a1_culane.py
# Predicting lane points for testing
python main_landec.py --train --mixed-precision --config configs/lane_detection/baseline/repvgg-a1_culane.py
# Testing with official scripts
./autotest_tusimple.sh repvgg-a1_baseline_culane test