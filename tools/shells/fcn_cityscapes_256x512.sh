#!/bin/bash
# Trained weights: fcn_cityscapes_256x512_20201226.pt
python main_semseg.py --train --config=configs/semantic_segmentation/fcn/resnet101_cityscapes_256x512.py --mixed-precision

# Val
python main_semseg.py --val --config=configs/semantic_segmentation/fcn/resnet101_cityscapes_256x512.py --mixed-precision
