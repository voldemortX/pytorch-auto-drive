#!/bin/bash
# Trained weights: erfnet_cityscapes_512x1024_20200918.pt
python main_semseg.py --epochs=150 --lr=0.0007 --batch-size=10 --dataset=city --model=erfnet --mixed-precision --exp-name=erfnet_cityscapes_512x1024
