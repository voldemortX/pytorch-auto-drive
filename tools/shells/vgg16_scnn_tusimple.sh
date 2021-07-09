#!/bin/bash
# Trained weights: vgg16_scnn_tusimple_20210224.pt
# Training
python main_landec.py --epochs=50 --lr=0.35 --batch-size=20 --dataset=tusimple --method=scnn --backbone=vgg16 --mixed-precision --exp-name=vgg16_scnn_tusimple
# Predicting lane points for testing
python main_landec.py --state=2 --batch-size=80 --continue-from=vgg16_scnn_tusimple.pt --dataset=tusimple --method=scnn --backbone=vgg16 --mixed-precision
# Testing with official scripts
./autotest_tusimple.sh vgg16_scnn_tusimple test
