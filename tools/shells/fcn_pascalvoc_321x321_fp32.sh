#!/bin/bash
# Trained weights: fcn_pascalvoc_321x321_fp32_20201111.pt
python main_semseg.py --train --config=configs/semantic_segmentation/fcn/resnet101_pascalvoc_321x321.py

# Val
python main_semseg.py --val --config=configs/semantic_segmentation/fcn/resnet101_pascalvoc_321x321.py
