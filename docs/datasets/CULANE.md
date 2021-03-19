# CULane  

## Prepare the dataset

1. The CULane dataset can be downloaded in their [official website](https://xingangpan.github.io/projects/CULane.html).

2. Change the `CULANE.BASE_DIR` in [configs.yaml](../../configs.yaml) to your dataset's location.

3. Pre-processing:

```
  cp -r <CULANE.BASE_DIR>/list/* <CULANE.BASE_DIR>/lists/
  python tools/culane_list_convertor.py
```

## Description

### Directory Structure

```
  <CULANE.BASE_DIR>
     ├─ driver_100_30frame    
     ├─ driver_161_90frame    
     ├─ driver_182_30frame    
     ├─ driver_193_90frame
     ├─ driver_23_30frame
     ├─ driver_37_30frame
     ├─ laneseg_label_w16
     ├─ list
     └─ lists
```

### Label Data Format

```
x1, y, x2, y-10, x3, y-20, ... , xn, y-10(n-1)
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
