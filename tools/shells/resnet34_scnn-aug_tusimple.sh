#!/bin/bash
# Trained weights: resnet34_scnn-aug_tusimple_20210723.pt
# Training
python -m torch.distributed.launch --nproc_per_node=2 --use_env main_landec.py --train --config=configs/lane_detection/scnn/resnet34_tusimple-aug.py
# Predicting lane points for testing
python main_landec.py --test --config=configs/lane_detection/scnn/resnet34_tusimple-aug.py
# Testing with official scripts
./autotest_tusimple.sh resnet34_scnn-aug_tusimple test checkpoints
