# Tusimple

## Prepare the dataset

1. The TuSimple dataset can be downloaded at their [github repo](https://github.com/TuSimple/tusimple-benchmark/issues/3). However, you'll also need [segmentation labels](https://drive.google.com/open?id=1LZDCnr79zuNH73NstZ8oIPDud0INCwb9), [list6_train.txt](https://github.com/cardwing/Codes-for-Lane-Detection/blob/master/ENet-TuSimple-Torch/list6/list6_train.txt), [list6_val.txt](https://github.com/cardwing/Codes-for-Lane-Detection/blob/master/ENet-TuSimple-Torch/list6/list6_val.txt) and [list_test.txt](https://github.com/cardwing/Codes-for-Lane-Detection/blob/master/ENet-TuSimple-Torch/list/list_test.txt) provided by [@cardwing](https://github.com/cardwing), thanks for their efforts.
2. Change the `TUSIMPLE.BASE_DIR` in [configs.yaml](../../configs.yaml) to your dataset's location.
3. Pre-processing:
   
First put the data lists you downloaded before in `TUSIMPLE.BASE_DIR/lists`. Then:

```
  python tools/tusimple_list_convertor.py
```

## Description

### Directory Structure

```
    <TUSIMPLE.BASE_DIR>
      ├─ clips
      ├─ lists
      ├─ segGT6
      ├─ label_data_0313.json
      ├─ label_data_0531.json
      ├─ label_data_0601.json
      └─ test_label.json
```

### Label Data Format

```
{
   'lanes': list. A list of lanes. For each list of one lane, the elements are width values on image.
   'h_samples': list. A list of height values corresponding to the 'lanes', which means len(h_samples) == len(lanes[i])
   'raw_file': str. 20th frame file path in a clip.
}
```

There will be at most 5 lane markings in `lanes`. The polylines are organized by the same distance gap (`h_sample` in each label data) from the recording car. It means you can pair each element in one lane and h_samples to get position of lane marking in images.

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
