#!/bin/bash
# Trained weights: resnet18_baseline_culane_20210222.pt
# Training
python main_landec.py --epochs=12 --lr=0.2 --batch-size=20 --dataset=culane --method=baseline --backbone=resnet18 --mixed-precision --exp-name=resnet18_baseline_culane
# Predicting lane points for testing
python main_landec.py --state=2 --batch-size=80 --continue-from=resnet18_baseline_culane.pt --dataset=culane --method=baseline --backbone=resnet18 --mixed-precision
# Testing with official scripts
./autotest_culane.sh resnet18_baseline_culane test
