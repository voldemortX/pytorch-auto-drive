#!/bin/bash
root=../../
data_dir=../../../../dataset/culane/
exp=$1
detect_dir=../../output/

# These can not be changed
w_lane=30;
iou=0.5;  # Set iou to 0.3 or 0.5
im_w=1640
im_h=590
list=${data_dir}list/val.txt
out=./output/${exp}_iou${iou}_validation.txt
python evaluate.py -a $data_dir -d $detect_dir -l $list -w $w_lane -t $iou -c $im_w -r $im_h -o $out

if ! [ -z "$2" ]
  then
    mkdir -p ../../${2}/${1}
    cp ${out} ../../${2}/${1}/val_result.txt
fi
