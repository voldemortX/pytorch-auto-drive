#!/bin/bash
# Trained weights: resnet18_scnn_tusimple_20210424.pt
# Training
python main_landec.py --epochs=50 --lr=0.2 --batch-size=20 --dataset=tusimple --method=scnn --backbone=resnet18 --mixed-precision --exp-name=resnet18_scnn_tusimple
# Predicting lane points for testing
python main_landec.py --state=2 --batch-size=80 --continue-from=resnet18_scnn_tusimple.pt --dataset=tusimple --method=scnn --backbone=resnet18 --mixed-precision --exp-name=resnet18_scnn_tusimple
# Testing with official scripts
./autotest_tusimple.sh resnet18_scnn_tusimple test
