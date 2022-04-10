#!/bin/bash
# Trained weights: erfnet_baseline_tusimple-aug_20210723.pt
# Training
python -m torch.distributed.launch --nproc_per_node=2 --use_env main_landet.py --train --config=configs/lane_detection/baseline/erfnet_tusimple_aug.py
# Predicting lane points for testing
python main_landet.py --test --config=configs/lane_detection/baseline/erfnet_tusimple.py
# Testing with official scripts
./autotest_tusimple.sh erfnet_baseline_tusimple-aug test checkpoints
