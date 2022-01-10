#!/bin/bash
# Trained weights: deeplabv3_voc_321x321_20201110.pt
python main_semseg.py --train --config=configs/semantic_segmentation/deeplabv3/resnet101_pascalvoc_321x321.py --mixed-precision

# Val
python main_semseg.py --val --config=configs/semantic_segmentation/deeplabv3/resnet101_pascalvoc_321x321.py --mixed-precision
