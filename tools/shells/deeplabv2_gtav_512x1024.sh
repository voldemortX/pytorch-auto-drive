#!/bin/bash
# Trained weights: deeplabv2_gtav_512x1024_20201223.pt
python main_semseg.py --epochs=10 --lr=0.002 --batch-size=4 --dataset=gtav --model=deeplabv2 --mixed-precision --exp-name=deeplabv2_gtav_512x1024
