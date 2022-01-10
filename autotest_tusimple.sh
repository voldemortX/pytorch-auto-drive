#!/bin/bash
data_dir=../../../../dataset/tusimple/
echo experiment name: $1
echo status: $2
echo save dir: $3

# Perform test/validation with official scripts
cd tools/tusimple_evaluation
if [ "$2" = "test" ]; then
    python lane.py ../../output/${1}.json ${data_dir}test_label.json $1 $3
else
    python lane.py ../../output/${1}.json ${data_dir}label_data_0531.json $1 $3
fi
cd ../../
