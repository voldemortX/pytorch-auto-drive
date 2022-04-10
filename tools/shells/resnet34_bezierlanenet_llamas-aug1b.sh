#!/bin/bash
# Trained weights: resnet34_bezierlanenet_llamas-aug1b_20211109.pt
# Training
python main_landet.py --train --config=configs/lane_detection/bezierlanenet/resnet34_llamas_aug1b.py --mixed-precision
# Predicting lane points for testing
python main_landet.py --val --config=configs/lane_detection/bezierlanenet/resnet34_llamas_aug1b.py --mixed-precision
# Testing with official scripts
./autotest_llamas.sh resnet34_bezierlanenet_llamas-aug1b val checkpoints
# Predict lane points for the eval server, find results in ./output
python main_landet.py --test --config=configs/lane_detection/bezierlanenet/resnet34_llamas_aug1b.py --mixed-precision
