#!/bin/bash
backend="cpp"  # or python
echo experiment name: $1
echo status: $2
echo save dir: $3

if [ "${backend}" = "cpp" ]; then
    cd tools/culane_evaluation
else
    echo Although python version works, we still suggest using the official Cpp version when possible!
    echo At least try them when you are writing a paper.
    echo Cpp and Python backends yield slightly different results.
    cd tools/culane_evaluation_py
fi

if [ "$2" = "test" ]; then
    # Perform test with official scripts
    ./eval.sh $1 $3
    # Calculate overall F1 score
    python cal_total.py --exp-name=$1 --save-dir=$3
else
    # Perform validation with official scripts
    ./eval_validation.sh $1 $3
fi
cd ../../
