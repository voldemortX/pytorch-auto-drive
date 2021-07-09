#!/bin/bash
# Trained weights: erfnet_scnn_tusimple_20210202.pt
# Training
python main_landec.py --epochs=50 --lr=0.2 --batch-size=20 --dataset=tusimple --method=scnn --backbone=erfnet --mixed-precision --exp-name=erfnet_scnn_tusimple
# Predicting lane points for testing
python main_landec.py --state=2 --batch-size=80 --continue-from=erfnet_scnn_tusimple.pt --dataset=tusimple --method=scnn --backbone=erfnet --mixed-precision
# Testing with official scripts
./autotest_tusimple.sh erfnet_scnn_tusimple test
