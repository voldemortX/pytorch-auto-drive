#!/bin/bash
# Trained weights: deeplabv2_pascalvoc_321x321_20201108.pt
python main_semseg.py --train --config=configs/semantic_segmentation/deeplabv2/resnet101_pascalvoc_321x321.py --mixed-precision

# Val
python main_semseg.py --val --config=configs/semantic_segmentation/deeplabv2/resnet101_pascalvoc_321x321.py --mixed-precision
