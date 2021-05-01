#!/bin/bash
echo experiment name: $1
echo status: $2
cd ./tools/llamas_evaluation/

if ["$2" = "val"]; then
    # we can provide the valid set to evaluate models
    python evaluate.py --pred_dir=../../output/valid --anno_dir=/home/guoshaohua/dataset/llamas/labels/valid --exp_name=$1
else
    echo "the test set of llamas is unavailable."
fi

cd ../../
