#!/bin/bash
echo experiment name: $1

# Perform testing with official scripts
cd tools/culane_evaluation
./eval.sh $1

# Calculate overall F1 score
python cal_total.py --exp-name=$1
cd ../../
