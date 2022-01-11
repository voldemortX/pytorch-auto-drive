#!/bin/bash
# Trained weights: vgg16_scnn_tusimple_20210224.pt
# Training
python main_landet.py --train --config=configs/lane_detection/scnn/vgg16_tusimple.py --mixed-precision
# Predicting lane points for testing
python main_landet.py --test --config=configs/lane_detection/scnn/vgg16_tusimple.py --mixed-precision
# Testing with official scripts
./autotest_tusimple.sh vgg16_scnn_tusimple test checkpoints
