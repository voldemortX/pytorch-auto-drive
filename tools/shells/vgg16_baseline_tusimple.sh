#!/bin/bash
# Trained weights: vgg16_baseline_tusimple_20210223.pt
# Training
python main_landec.py --epochs=50 --lr=0.25 --batch-size=20 --dataset=tusimple --method=baseline --backbone=vgg16 --mixed-precision --exp-name=vgg16_baseline_tusimple
# Predicting lane points for testing
python main_landec.py --state=2 --batch-size=80 --continue-from=vgg16_baseline_tusimple.pt --dataset=tusimple --method=baseline --backbone=vgg16 --mixed-precision --exp-name=vgg16_baseline_tusimple
# Testing with official scripts
./autotest_tusimple.sh vgg16_baseline_tusimple test
