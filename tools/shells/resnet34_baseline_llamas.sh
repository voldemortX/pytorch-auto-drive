#!/bin/bash
# Trained weights: resnet34_baseline_llamas_20210625.pt
# Training
python main_landet.py --train --config=configs/lane_detection/baseline/resnet34_llamas.py --mixed-precision
# Predicting lane points for testing
python main_landet.py --test --config=configs/lane_detection/baseline/resnet34_llamas.py --mixed-precision
# Testing with official scripts
./autotest_llamas.sh resnet34_baseline_llamas test checkpoints
