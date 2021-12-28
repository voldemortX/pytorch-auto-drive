#!/bin/bash
# Trained weights: repvgg-a2_baseline_culane_20211228.pt
# Training
python main_landec.py --train --mixed-precision --config configs/lane_detection/baseline/repvgg-a2_culane.py
# Predicting lane points for testing
python main_landec.py --train --mixed-precision --config configs/lane_detection/baseline/repvgg-a2_culane.py
# Testing with official scripts
./autotest_tusimple.sh repvgg-a2_baseline_culane test