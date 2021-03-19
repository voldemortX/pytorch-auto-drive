# PASCAL VOC 2012

## Prepare the dataset

1. The PASCAL VOC 2012 dataset we use is the commonly used 10582 training set version. If you don't already have that dataset, we refer you to [Google](https://www.google.com) or this [blog](https://www.sun11.me/blog/2018/how-to-use-10582-trainaug-images-on-DeeplabV3-code/).

2. Change the `PASCAL_VOC.BASE_DIR` in [configs.yaml](../../configs.yaml) to your dataset's location.

## Description

### Directory Structure

```
    ├── <PASCAL_VOC.BASE_DIR>/VOCtrainval_11-May-2012/VOCdevkit/VOC2012                    
        ├── Annotations 
        ├── ImageSets
        │   ├── Segmentation
        │   └── ... 
        ├── JPEGImages
        ├── SegmentationClass
        ├── SegmentationClassAug
        └── ...
```

*More details are coming soon.*
