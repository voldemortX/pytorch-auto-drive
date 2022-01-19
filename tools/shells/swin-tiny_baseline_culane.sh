#!/bin/bash
# Trained weights: swin-tiny_baseline_culane_20220119.pt
# Training
python -m torch.distributed.launch --nproc_per_node=2 --use_env main_landec.py --train --config=configs/lane_detection/baseline/swin-tiny_culane.py
# Predicting lane points for testing
python main_landec.py --test --config=configs/lane_detection/baseline/swin-tiny_culane.py
# Testing with official scripts
./autotest_culane.sh swin-tiny_baseline_culane test checkpoints
