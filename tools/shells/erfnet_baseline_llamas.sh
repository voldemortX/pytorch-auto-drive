#!/bin/bash
# Trained weights: erfnet_baseline_llamas_20210625.pt
# Training
python main_landec.py --train --config=configs/lane_detection/baseline/erfnet_llamas.py --mixed-precision
# Predicting lane points for testing
python main_landec.py --val --config=configs/lane_detection/baseline/erfnet_llamas.py --mixed-precision
# Testing with official scripts
./autotest_llamas.sh erfnet_baseline_llamas val checkpoints
# Predict lane points for the eval server, find results in ./output
python main_landec.py --test --config=configs/lane_detection/baseline/erfnet_llamas.py --mixed-precision
