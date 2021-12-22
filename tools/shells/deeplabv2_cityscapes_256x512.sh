#!/bin/bash
# Trained weights: deeplabv2_cityscapes_256x512_20201225.pt
python main_semseg.py --train --config=configs/semantic_segmentation/deeplabv2/resnet101_cityscapes_256x512.py --mixed-precision

# Val
python main_semseg.py --val --config=configs/semantic_segmentation/deeplabv2/resnet101_cityscapes_256x512.py --mixed-precision
