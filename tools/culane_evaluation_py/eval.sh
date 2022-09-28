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
list0=${data_dir}list/test_split/test0_normal.txt
list1=${data_dir}list/test_split/test1_crowd.txt
list2=${data_dir}list/test_split/test2_hlight.txt
list3=${data_dir}list/test_split/test3_shadow.txt
list4=${data_dir}list/test_split/test4_noline.txt
list5=${data_dir}list/test_split/test5_arrow.txt
list6=${data_dir}list/test_split/test6_curve.txt
list7=${data_dir}list/test_split/test7_cross.txt
list8=${data_dir}list/test_split/test8_night.txt
out0=./output/out0_normal.txt
out1=./output/out1_crowd.txt
out2=./output/out2_hlight.txt
out3=./output/out3_shadow.txt
out4=./output/out4_noline.txt
out5=./output/out5_arrow.txt
out6=./output/out6_curve.txt
out7=./output/out7_cross.txt
out8=./output/out8_night.txt
python evaluate.py -a $data_dir -d $detect_dir -l $list0 -w $w_lane -t $iou -c $im_w -r $im_h -o $out0
python evaluate.py -a $data_dir -d $detect_dir -l $list1 -w $w_lane -t $iou -c $im_w -r $im_h -o $out1
python evaluate.py -a $data_dir -d $detect_dir -l $list2 -w $w_lane -t $iou -c $im_w -r $im_h -o $out2
python evaluate.py -a $data_dir -d $detect_dir -l $list3 -w $w_lane -t $iou -c $im_w -r $im_h -o $out3
python evaluate.py -a $data_dir -d $detect_dir -l $list4 -w $w_lane -t $iou -c $im_w -r $im_h -o $out4
python evaluate.py -a $data_dir -d $detect_dir -l $list5 -w $w_lane -t $iou -c $im_w -r $im_h -o $out5
python evaluate.py -a $data_dir -d $detect_dir -l $list6 -w $w_lane -t $iou -c $im_w -r $im_h -o $out6
python evaluate.py -a $data_dir -d $detect_dir -l $list7 -w $w_lane -t $iou -c $im_w -r $im_h -o $out7
python evaluate.py -a $data_dir -d $detect_dir -l $list8 -w $w_lane -t $iou -c $im_w -r $im_h -o $out8
cat ./output/out*.txt>./output/${exp}_iou${iou}_split.txt

if ! [ -z "$2" ]
  then
    mkdir -p ../../${2}/${1}
    cp ./output/${exp}_iou${iou}_split.txt ../../${2}/${1}/test_result.txt
fi
