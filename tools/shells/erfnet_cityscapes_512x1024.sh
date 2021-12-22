#!/bin/bash
# Trained weights: erfnet_cityscapes_512x1024_20200918.pt
python main_semseg.py --train --config=configs/semantic_segmentation/erfnet/cityscapes_512x1024.py --mixed-precision

# Val
python main_semseg.py --val --config=configs/semantic_segmentation/erfnet/cityscapes_512x1024.py --mixed-precision
