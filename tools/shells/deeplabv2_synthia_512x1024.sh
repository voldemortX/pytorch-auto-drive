#!/bin/bash
# Trained weights: deeplabv2_synthia_512x1024_20201225.pt
python main_semseg.py --train --config=configs/semantic_segmentation/deeplabv2/resnet101_synthia_512x1024.py --mixed-precision

# Val
python main_semseg.py --val --config=configs/semantic_segmentation/deeplabv2/resnet101_synthia_512x1024.py --mixed-precision
