#!/bin/bash
echo experiment name: $1
echo status: $2

cd tools/culane_evaluation
if [ "$2" = "test" ]; then
    # Perform test with official scripts
    ./eval.sh $1
    # Calculate overall F1 score
    python cal_total.py --exp-name=$1
else
    # Perform validation with official scripts
    ./eval_validation.sh $1
fi
cd ../../
