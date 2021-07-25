#!/bin/bash
# Trained weights: enet_cityscapes_512x1024_20210219.pt
# Step-1: Pre-train encoder
python main_semseg.py --epochs=300 --lr=0.0008 --batch-size=16 --weight-decay=0.0002 --dataset=city --model=enet --mixed-precision --encoder-only --exp-name=enet_cityscapes_512x1024_encoder
# Step-2: Train the entire network
python main_semseg.py --state=2  --continue-from=enet_cityscapes_512x1024_encoder.pt --epochs=300 --lr=0.0008 --batch-size=16 --weight-decay=0.0002 --dataset=city --model=enet --mixed-precision --exp-name=enet_cityscapes_512x1024
