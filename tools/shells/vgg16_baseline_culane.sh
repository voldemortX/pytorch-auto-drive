#!/bin/bash
# Trained weights: vgg16_baseline_culane_20210309.pt
# Training
python main_landec.py --epochs=12 --lr=0.2 --batch-size=20 --dataset=culane --method=baseline --backbone=vgg16 --mixed-precision --exp-name=vgg16_baseline_culane
# Predicting lane points for testing
python main_landec.py --state=2 --batch-size=80 --continue-from=vgg16_baseline_culane.pt --dataset=culane --method=baseline --backbone=vgg16 --mixed-precision
# Testing with official scripts
./autotest_culane.sh vgg16_baseline_culane test
