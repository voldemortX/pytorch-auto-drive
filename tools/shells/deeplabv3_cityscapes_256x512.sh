#!/bin/bash
# Trained weights: deeplabv3_city_256x512_20201226.pt
python main_semseg.py --train --config=configs/semantic_segmentation/deeplabv3/resnet101_cityscapes_256x512.py --mixed-precision

# Val
python main_semseg.py --val --config=configs/semantic_segmentation/deeplabv3/resnet101_cityscapes_256x512.py --mixed-precision
