#!/bin/bash
# TODO try multi-GPU training
# Training
python -m torch.distributed.launch --nproc_per_node=2 --use_env main_landet.py --train --config=configs/lane_detection/resa/erfnet_tusimple-aug.py
# Predicting lane points for testing
python main_landet.py --test --config=configs/lane_detection/resa/erfnet_tusimple.py
# Testing with official scripts
./autotest_tusimple.sh erfnet_resa_tusimple-aug test checkpoints
