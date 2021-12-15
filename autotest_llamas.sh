#!/bin/bash
echo experiment name: $1
echo status: $2
echo save dir: $3
data_dir=../../../../dataset/llamas/labels/valid
cd tools/llamas_evaluation/

if [ "$2" = "val" ]; then
    # we can provide the valid set to evaluate models
    python evaluate.py --pred_dir=../../output/valid --anno_dir=${data_dir} --exp_name=$1 --save-dir=$3
else
    echo "The test set of llamas is not public available."
fi
cd ../../