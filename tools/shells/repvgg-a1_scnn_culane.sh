#!/bin/bash
# Trained weights: repvgg-a1_scnn_culane_20220112.pt
# Training
python -m torch.distributed.launch --nproc_per_node=2 --use_env main_landet.py --train --mixed-precision --config configs/lane_detection/scnn/repvgg-a1_culane.py
# Predicting lane points for testing
python main_landet.py --test --config configs/lane_detection/scnn/repvgg-a1_culane.py --mixed-precision
# Testing with official scripts
./autotest_culane.sh repvgg-a1_scnn_culane test
