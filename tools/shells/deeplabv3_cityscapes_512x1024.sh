#!/bin/bash
# Trained weights: deeplabv3_cityscapes_512x1024_20210322.pt
python main_semseg.py --train --config=configs/semantic_segmentation/deeplabv3/resnet101_cityscapes_512x1024.py --mixed-precision

# Val
python main_semseg.py --val --config=configs/semantic_segmentation/deeplabv3/resnet101_cityscapes_512x1024.py --mixed-precision
