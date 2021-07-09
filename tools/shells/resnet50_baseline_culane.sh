#!/bin/bash
# Trained weights: resnet50_baseline_culane_20210308.pt
# Training, scale lr linearly on 11G GPU (square root scaling does not converge on this dataset)
python main_landec.py --epochs=12 --lr=0.08 --batch-size=8 --dataset=culane --method=baseline --backbone=resnet50 --workers=4 --warmup-steps=500 --mixed-precision --exp-name=resnet50_baseline_culane
# Predicting lane points for testing
python main_landec.py --state=2 --batch-size=32 --continue-from=resnet50_baseline_culane.pt --dataset=culane --method=baseline --backbone=resnet50 --workers=4 --mixed-precision
# Testing with official scripts
./autotest_culane.sh resnet50_baseline_culane test
