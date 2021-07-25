#!/bin/bash
# Trained weights: resnet18s_lstr-aug_tusimple_20210629.pt
# Training
python main_landec.py --epochs=2000 --lr=0.00025 --batch-size=20 --dataset=tusimple --method=lstr --backbone=resnet18s --workers=16 --aug --exp-name=resnet18s_lstr-aug
# Predicting lane points for testing
python main_landec.py --state=2 --batch-size=80 --continue-from=resnet18s_lstr-aug.pt --dataset=tusimple --method=lstr --backbone=resnet18s --exp-name=resnet18s_lstr-aug
# Testing with official scripts
./autotest_tusimple.sh resnet18s_lstr-aug_tusimple test
