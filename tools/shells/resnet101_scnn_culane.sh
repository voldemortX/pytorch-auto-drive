#!/bin/bash
# Trained weights: resnet101_scnn_culane_20210314.pt
# Training, scale lr linearly on 11G GPU (square root scaling does not converge on this dataset)
python main_landec.py --epochs=12 --lr=0.08 --batch-size=8 --dataset=culane --method=scnn --backbone=resnet101 --workers=4 --warmup-steps=500 --mixed-precision --exp-name=resnet101_scnn_culane
# Predicting lane points for testing
python main_landec.py --state=2 --batch-size=32 --continue-from=resnet101_scnn_culane.pt --dataset=culane --method=scnn --backbone=resnet101 --workers=4 --mixed-precision
# Testing with official scripts
./autotest_culane.sh resnet101_scnn_culane test
