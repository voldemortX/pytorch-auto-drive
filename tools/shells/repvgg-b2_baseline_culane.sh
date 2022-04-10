#!/bin/bash
# Trained weights: repvgg-b2_baseline_culane_20220112.pt
# Training
python -m torch.distributed.launch --nproc_per_node=2 --use_env main_landet.py --train --mixed-precision --config configs/lane_detection/baseline/repvgg_b2_culane.py
# Predicting lane points for testing
python main_landet.py --test --config configs/lane_detection/baseline/repvgg_b2_culane.py --mixed-precision
# Testing with official scripts
./autotest_culane.sh repvgg-b2_baseline_culane test
