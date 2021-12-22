#!/bin/bash
# Trained weights: deeplabv2_cityscapes_512x1024_20201219.pt
python main_semseg.py --train --config=configs/semantic_segmentation/deeplabv2/resnet101_cityscapes_512x1024.py --mixed-precision

# Val
python main_semseg.py --val --config=configs/semantic_segmentation/deeplabv2/resnet101_cityscapes_512x1024.py --mixed-precision
