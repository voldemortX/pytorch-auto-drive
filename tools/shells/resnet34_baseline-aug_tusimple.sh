#!/bin/bash
# Trained weights: resnet34_baseline-aug_tusimple_20210723.pt
# Training
python -m torch.distributed.launch --nproc_per_node=2 --use_env main_landet.py --train --config=configs/lane_detection/baseline/resnet34_tusimple_aug.py --mixed-precision
# Predicting lane points for testing
python main_landet.py --test --config=configs/lane_detection/baseline/resnet34_tusimple_aug.py --mixed-precision
# Testing with official scripts
./autotest_tusimple-aug.sh resnet34_baseline_tusimple-aug test checkpoints
