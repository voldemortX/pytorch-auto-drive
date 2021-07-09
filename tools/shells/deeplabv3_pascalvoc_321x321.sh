#!/bin/bash
# Trained weights: deeplabv3_voc_321x321_20201110.pt
python main_semseg.py --epochs=30 --lr=0.002 --batch-size=8 --dataset=voc --model=deeplabv3 --workers=4 --mixed-precision --exp-name=deeplabv3_voc_321x321
