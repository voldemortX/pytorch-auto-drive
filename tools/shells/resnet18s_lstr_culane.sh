#!/bin/bash
# Trained weights: resnet18s_lstr_culane_20210722.pt
exp_name=resnet18s_lstr_culane
# Training
python main_landec.py --epochs=150 --lr=0.00025 --batch-size=20 --workers=16 --dataset=culane --method=lstr --backbone=resnet18s --exp-name=${exp_name}
# Predicting lane points for testing
python main_landec.py --state=2 --batch-size=80 --continue-from=${exp_name}.pt --dataset=culane --method=lstr --backbone=resnet18s --exp-name=${exp_name}
# Testing with official scripts
./autotest_culane.sh ${exp_name} test
