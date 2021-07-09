#!/bin/bash
# Trained weights: deeplabv2_pascalvoc_321x321_20201108.pt
python main_semseg.py --epochs=30 --lr=0.002 --batch-size=8 --dataset=voc --model=deeplabv2 --workers=4 --mixed-precision --exp-name=deeplabv2_pascalvoc_321x321
