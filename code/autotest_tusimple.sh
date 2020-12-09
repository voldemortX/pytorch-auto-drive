#!/bin/bash
data_dir=../../../../../dataset/tusimple/
echo experiment name: $1

# Perform testing with official scripts
cd tools/tusimple_evaluation
python lane.py ../../output/tusimple_pred.json ${data_dir}test_label.json $1
cd ../../
