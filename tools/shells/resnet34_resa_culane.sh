#!/bin/bash
# Trained weights: resnet34_resa_culane_20211016.pt
exp_name=resnet34_resa_culane
url=tcp://localhost:12345
# Training
python -m torch.distributed.launch --nproc_per_node=4 --use_env main_landec.py --epochs=12 --lr=0.06 --warmup-steps=500 --batch-size=5 --workers=4 --dataset=culane --method=resa --backbone=resnet34 --world-size=4 --dist-url=${url} --exp-name=${exp_name}
# Predicting lane points for testing
python main_landec.py --state=2 --batch-size=20 --continue-from=${exp_name}.pt --dataset=culane --method=resa --backbone=resnet34 --exp-name=${exp_name}
# Testing with official scripts
./autotest_culane.sh ${exp_name} test
