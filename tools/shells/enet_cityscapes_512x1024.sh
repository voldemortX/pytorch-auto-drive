#!/bin/bash
# Trained weights: enet_cityscapes_512x1024_20210219.pt
# Step-1: Pre-train encoder
python main_semseg.py --train --config=configs/semantic_segmentation/enet/cityscapes_512x1024_encoder.py --mixed-precision

# Step-2: Train the entire network
python main_semseg.py --train --config=configs/semantic_segmentation/enet/cityscapes_512x1024.py --mixed-precision

# Val
python main_semseg.py --val --config=configs/semantic_segmentation/enet/cityscapes_512x1024.py --mixed-precision
