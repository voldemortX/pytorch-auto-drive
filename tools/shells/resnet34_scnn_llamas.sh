#!/bin/bash
# Trained weights: resnet34_scnn_llamas_20210625.pt
# Training
python main_landec.py --epochs=18 --lr=0.1 --batch-size=20 --dataset=llamas --method=scnn --backbone=resnet34 --mixed-precision --exp-name=resnet34_scnn_llamas
# Predicting lane points for testing
python main_landec.py --state=3 --batch-size=80 --continue-from=resnet34_scnn_llamas.pt --dataset=llamas --method=scnn --backbone=resnet34 --mixed-precision
# Testing with official scripts
./autotest_llamas.sh resnet34_scnn_llamas val