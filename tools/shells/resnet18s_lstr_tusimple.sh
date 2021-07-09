#!/bin/bash
# Trained weights: resnet18s_lstr_tusimple_20210701.pt
# Training
python main_landec.py --epochs=2000 --lr=0.00025 --batch-size=20 --dataset=tusimple --method=lstr --backbone=resnet18s --workers=16 --exp-name=resnet18s_lstr
# Predicting lane points for testing
python main_landec.py --state=2 --batch-size=80 --continue-from=resnet18s_lstr.pt --dataset=tusimple --method=lstr --backbone=resnet18s
# Testing with official scripts
./autotest_tusimple.sh resnet18s_lstr_tusimple test
