#!/bin/bash
# Trained weights: resnet34_scnn_tusimple_20210424.pt
# Training
python main_landec.py --epochs=50 --lr=0.2 --batch-size=20 --dataset=tusimple --method=scnn --backbone=resnet34 --mixed-precision --exp-name=resnet34_scnn_tusimple
# Predicting lane points for testing
python main_landec.py --state=2 --batch-size=80 --continue-from=resnet34_scnn_tusimple.pt --dataset=tusimple --method=scnn --backbone=resnet34 --mixed-precision
# Testing with official scripts
./autotest_tusimple.sh resnet34_scnn_tusimple test
