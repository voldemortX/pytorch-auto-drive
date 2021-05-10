# Datasets

## Basic information of lane detection datasets

| Name | #Train | #Validation | #Test | Resolution (H x W) | #Lanes | |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| CULane | 88880 | 9675 | 34680 | 590 x 1640 | <=4 | [instruction](./datasets/CULANE.md)|
| TuSimple | 3268 | 358 | 2782 | 720 x 1280 | <=5 | [instruction](./datasets/TUSIMPLE.md) |
| LLAMAS* | 58269 | 20844 | 20929 | 717 x 1276 | <= 4 | [instruction](./datasets/LLAMAS.md) |

*\* The number of lanes in llamas dataset is more than 4, but most methods & evaluation metrics only use 4 lanes.*

## Basic information of semantic segmentation datasets

| Name | #Train | #Validation | #Test | Resolution (H x W) | #Classes | |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| PASCAL VOC 2012* | 10582 | 1449 | - | various | 21 | [instruction](./datasets/PASCALVOC.md) |
| Cityscapes | 2975 | 500 | - | 1024 x 2048 | 19 | [instruction](./datasets/CITYSCAPES.md) |
| GTAV | 24966 | - | - | mostly 1052 x 1914 | 19 | [instruction](./datasets/GTAV.md) |
| SYNTHIA** | 9400 | - | - | 760 x 1280 | 23 | [instruction](./datasets/SYNTHIA.md) |

*\* Extended by SBD.*

*\*\* SYNTHIA-RAND-CITYSCAPES.*

*- Not used or label not available.*
