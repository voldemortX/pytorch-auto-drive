# Datasets

## Basic information of two lane detection datasets

| Name | #Frame | #Train | #Validation | #Test | Resolution | # Lane > 5 |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | 
| CULane | 133235 | 88880 | 9675 |34680 | 1640 x 590 | x |
| Tusimple | 6408 | 3268 | 358 |2782 | 1280 x 720 | x |


## CULane  

[Download Page](https://xingangpan.github.io/projects/CULane.html)

### Directory Structure

```
  CULane_path
     ├─ driver_100_30frame    
     ├─ driver_161_90frame    
     ├─ driver_182_30frame    
     ├─ driver_193_90frame
     ├─ driver_23_30frame
     ├─ driver_37_30frame
     ├─ laneseg_label_w16
     └─ list
```

### Label Data Format

```
x<sub>1</sub>, y, x<sub>2</sub>, y+10, x<sub>3</sub>, y+20 , ... , x<sub>n</sub>, y+10(n-1)
```

For each image, there would be a .txt annotation file, in which each line gives the x,y coordinates for key points of a lane marking. The CULane dataset, focus attention on the detection of four lane markings, which are paid most attention to in real applications.
 
For example,

```
-20.4835 580 19.3893 570 58.1682 560 98.1783 550 137.929 540 177.709 530 216.495 520 256.512 510 296.276 500 336.008 490 375.78 480 415.941 470 456.696 460 496.456 450 537.226 440 577.47 430 618.252 420 659.177 410 
532.893 590 542.567 580 553.704 570 564.84 560 575.977 550 587.139 540 598.302 530 609.465 520 620.628 510 631.944 500 643.107 490 654.27 480 665.432 470 676.595 460 687.912 450 699.075 440 710.237 430 721.4 420 732.563 410 
1170.27 590 1151.37 580 1130.68 570 1110.6 560 1089.91 550 1068.95 540 1048.26 530 1027.56 520 1007.77 510 986.81 500 966.115 490 945.788 480 925.092 470 904.506 460 883.81 450 863.13 440 842.434 430 821.739 420 801.059 410 
1679.87 560 1626.23 550 1574.15 540 1520.62 530 1467.55 520 1414.57 510 1361.5 500 1307.8 490 1255.24 480 1202.18 470 1149.46 460 1096.4 450 1042.63 440 990.059 430 936.993 420 884.22 410 
```
*Each row in xxx.txt represents a lane mark.*

Training/validation/testing list:

For train_gt.txt, which is used for training.

```
input image   per-pixel label   four 0/1 numbers which indicate the existance of four lane markings from left to right
```

For example,

```
/driver_23_30frame/05151649_0422.MP4/00000.jpg /laneseg_label_w16/driver_23_30frame/05151649_0422.MP4/00000.png 1 1 1 1
/driver_23_30frame/05151649_0422.MP4/00300.jpg /laneseg_label_w16/driver_23_30frame/05151649_0422.MP4/00300.png 1 1 1 1
...
/driver_23_30frame/05151649_0422.MP4/00330.jpg /laneseg_label_w16/driver_23_30frame/05151649_0422.MP4/00330.png 1 1 1 1
```


## Tusimple

[Download Page](https://github.com/TuSimple/tusimple-benchmark/issues/3)

### Directory Structure

```
    Tusimple_path
      ├─ clips
      ├─ segGT
      ├─ segGT6
      ├─ label_data_0313.json
      ├─ label_data_0531.json
      ├─ label_data_0601.json
      ├─ test_label.json
      ├─ test_baseline.json
      └─ test_label_0627.json
```

### Label Data Format

```
{
   'lanes': list. A list of lanes. For each list of one lane, the elements are width values on image.
   'h_samples': list. A list of height values corresponding to the 'lanes', which means len(h_samples) == len(lanes[i])
   'raw_file': str. 20th frame file path in a clip.
}
```

There will be at most lane markings in `lanes`. The tusimple dataset expects at most 4 lane markings, current lane and left/right lanes that are essential for the control of the car. The polylines are organized by the same distance gap (`h_sample` in each label data) from the recording car. It means you can pair each element in one lane and h_samples to get position of lane marking in images.

For example,

```
{
  "lanes": [
        [-2, -2, -2, -2, 632, 625, 617, 609, 601, 594, 586, 578, 570, 563, 555, 547, 539, 532, 524, 516, 508, 501, 493, 485, 477, 469, 462, 454, 446, 438, 431, 423, 415, 407, 400, 392, 384, 376, 369, 361, 353, 345, 338, 330, 322, 314, 307, 299],
        [-2, -2, -2, -2, 719, 734, 748, 762, 777, 791, 805, 820, 834, 848, 863, 877, 891, 906, 920, 934, 949, 963, 978, 992, 1006, 1021, 1035, 1049, 1064, 1078, 1092, 1107, 1121, 1135, 1150, 1164, 1178, 1193, 1207, 1221, 1236, 1250, 1265, -2, -2, -2, -2, -2],
        [-2, -2, -2, -2, -2, 532, 503, 474, 445, 416, 387, 358, 329, 300, 271, 241, 212, 183, 154, 125, 96, 67, 38, 9, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2],
        [-2, -2, -2, 781, 822, 862, 903, 944, 984, 1025, 1066, 1107, 1147, 1188, 1229, 1269, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2]
       ],
  "h_samples": [240, 250, 260, 270, 280, 290, 300, 310, 320, 330, 340, 350, 360, 370, 380, 390, 400, 410, 420, 430, 440, 450, 460, 470, 480, 490, 500, 510, 520, 530, 540, 550, 560, 570, 580, 590, 600, 610, 620, 630, 640, 650, 660, 670, 680, 690, 700, 710],
  "raw_file": "path_to_clip"
}
```
*`-2` in `lanes` means on some h_sample, there is no exsiting lane marking. The first existing point in the first lane is (632, 280).*
