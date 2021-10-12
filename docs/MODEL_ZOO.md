# Welcome to pytorch-auto-drive model zoo

## Lane detection performance:

| method | backbone | data<br>augmentation | resolution | mixed precision? | dataset | metric | average | best | training time <br> *(2080 Ti)* |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Baseline | VGG16 | simple | 360 x 640 | *yes* | TuSimple | Accuracy | 93.79% | 93.94% | 1.5h |
| Baseline | ResNet18 | simple | 360 x 640 | *yes* | TuSimple | Accuracy | 94.18% | 94.25% | 0.7h |
| Baseline | ResNet34 | simple | 360 x 640 | *yes* | TuSimple | Accuracy | 95.23% | 95.31% | 1.1h |
| Baseline | ResNet34 | strong | 360 x 640 | *no* | TuSimple | Accuracy | 92.14% | 92.68% | 1.2h* |
| Baseline | ResNet50 | simple | 360 x 640 | *yes* | TuSimple | Accuracy | 95.07% | 95.12% | 1.5h |
| Baseline | ResNet101 | simple | 360 x 640 | *yes* | TuSimple | Accuracy | 95.15% | 95.19% | 2.6h |
| Baseline | ERFNet | simple | 360 x 640 | *yes* | TuSimple | Accuracy | 96.02% | 96.04% | 0.8h |
| Baseline | ERFNet | strong | 360 x 640 | *no* | TuSimple | Accuracy | 94.21% | 94.37% | 0.9h* |
| Baseline | ENet<sup>#</sup> | simple | 360 x 640 | *yes* | TuSimple | Accuracy | 95.55% | 95.61% | 1h<sup>+</sup> |
| SCNN | VGG16 | simple | 360 x 640 | *yes* | TuSimple | Accuracy | 95.01% | 95.17% | 2h |
| SCNN | ResNet18 | simple | 360 x 640 | *yes* | TuSimple | Accuracy | 94.69% | 94.77% | 1.2h |
| SCNN | ResNet34 | simple | 360 x 640 | *yes* | TuSimple | Accuracy | 95.19% | 95.25% | 1.6h |
| SCNN | ResNet34 | strong | 360 x 640 | *no* | TuSimple | Accuracy | 92.62% | 93.42% | 1.7h* |
| SCNN | ResNet50 | simple | 360 x 640 | *yes* | TuSimple | Accuracy | 95.43% | 95.56% | 2.4h |
| SCNN | ResNet101 | simple | 360 x 640 | *yes* | TuSimple | Accuracy | 95.56% | 95.69% | 3.5h |
| SCNN | ERFNet | simple | 360 x 640 | *yes* | TuSimple | Accuracy | 96.18% | 96.29% | 1.6h |
| SCNN | ERFNet | strong | 360 x 640 | *no* | TuSimple | Accuracy | 95.00% | 95.26% | 1.7h* |
| LSTR | ResNet18s<sup>#</sup> | simple | 360 x 640 | *no* | TuSimple | Accuracy | 91.91% | 92.40% | 14.2h |
| LSTR | ResNet18s<sup>#</sup> | strong | 360 x 640 | *no* | TuSimple | Accuracy | 94.91% | 95.06% | 15.5h |
| Baseline | VGG16 | simple | 288 x 800 | *yes* | CULane | F measure | 65.93 | 66.09 | 9.3h |
| Baseline | ResNet18 | simple | 288 x 800 | *yes* | CULane | F measure | 65.19 | 65.30 | 5.3h |
| Baseline | ResNet34 | simple | 288 x 800 | *yes* | CULane | F measure | 69.82 | 69.92 | 7.3h |
| Baseline | ResNet50 | simple | 288 x 800 | *yes* | CULane | F measure | 68.31 | 68.48 | 12.4h |
| Baseline | ResNet101 | simple | 288 x 800 | *yes* | CULane | F measure | 71.29 | 71.37 | 20.0h |
| Baseline | ERFNet | simple | 288 x 800 | *yes* | CULane | F measure | 73.40 | 73.49 | 6h |
| Baseline | ENet<sup>#</sup> | simple | 288 x 800 | *yes* | CULane | F measure | 69.39 | 69.90 | 6.4h<sup>+</sup> |
| SCNN | VGG16 | simple | 288 x 800 | *yes* | CULane | F measure | 74.02 | 74.29 | 12.8h |
| SCNN | ResNet18 | simple | 288 x 800 | *yes* | CULane | F measure | 71.94 | 72.19 | 8.0h |
| SCNN | ResNet34 | simple | 288 x 800 | *yes* | CULane | F measure | 72.44 | 72.70 | 10.7h |
| SCNN | ResNet50 | simple | 288 x 800 | *yes* | CULane | F measure | 72.95 | 73.03 | 17.9h |
| SCNN | ResNet101 | simple | 288 x 800 | *yes* | CULane | F measure | 73.29 | 73.58 | 25.7h |
| SCNN | ERFNet | simple | 288 x 800 | *yes* | CULane | F measure | 73.85 | 74.03 | 11.3h |
| LSTR | ResNet18s-2X<sup>#</sup> | simple | 288 x 800 | *no* | CULane | F measure | 35.72 | 39.17 | 28.5h* |
| LSTR | ResNet18s-2X<sup>#</sup> | strong | 288 x 800 | *no* | CULane | F measure | 66.85 | 67.21 | 31.5h* |
| LSTR | ResNet34 | strong | 288 x 800 | *no* | CULane | F measure | 71.16 | 71.52 | 70h* |
| Baseline | ERFNet | simple | 360 x 640 | *yes* | LLAMAS | F measure | 95.94 | 96.13 | 10.9h<sup>+</sup> |
| Baseline | VGG16 | simple | 360 x 640 | *yes* | LLAMAS | F measure | 95.05 | 95.11 | 9.3h |
| Baseline | ResNet34 | simple | 360 x 640 | *yes* | LLAMAS | F measure | 95.90 | 95.91 | 7.0h |
| SCNN | ERFNet | simple | 360 x 640 | *yes* | LLAMAS | F measure | 95.89 | 95.94 | 14.2h<sup>+</sup> |
| SCNN | VGG16 | simple | 360 x 640 | *yes* | LLAMAS | F measure | 96.39 | 96.42 | 12.5h |
| SCNN | ResNet34 | simple | 360 x 640 | *yes* | LLAMAS | F measure | 96.17 | 96.19 | 10.1h |

*All performance is measured with ImageNet pre-training and reported as 3 times average/best on test set.*

*The test set annotations of LLAMAS are not public, so we provide validation set result in this table.*

*<sup>+</sup> Measured on a single GTX 1080Ti.*

*<sup>#</sup> No pre-training.*

*\* Trained on a 1080 Ti cluster, with CUDA 9.0 PyTorch 1.3, training time is estimated as: single 2080 Ti, mixed precision.*

### TuSimple detailed performance (best):

| method | backbone | data<br>augmentation | accuracy | FP | FN | |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Baseline | VGG16 | simple | 93.94% | 0.0998 | 0.1021 | [model](https://drive.google.com/file/d/1ChK0hApqLU0xUiEm4Wul-gNYQDQka151/view?usp=sharing) \| [shell](../tools/shells/vgg16_baseline_tusimple.sh) |
| Baseline | ResNet18 | simple | 94.25% | 0.0881 | 0.0894 | [model](https://drive.google.com/file/d/17VKnwsN4WMbpnD4DgaaerppjXybqn-LG/view?usp=sharing) \| [shell](../tools/shells/resnet18_baseline_tusimple.sh) |
| Baseline | ResNet34 | simple | 95.31% | 0.0640 | 0.0622 | [model](https://drive.google.com/file/d/1NAck0aQZK_wAHer4xB8xzegxDWk9EFtG/view?usp=sharing) \| [shell](../tools/shells/resnet34_baseline_tusimple.sh) |
| Baseline | ResNet34 | strong | 92.68% | 0.1073 | 0.1221 | [model](https://drive.google.com/file/d/1OhN2tWIep9ncKFf-_2RqUEaSJvPK60cn/view?usp=sharing) \| [shell](../tools/shells/resnet34_baseline-aug_tusimple.sh) |
| Baseline | ResNet50 | simple | 95.12% | 0.0649 | 0.0653 | [model](https://drive.google.com/file/d/10KBMVGc63kPvqL_2deaLfTfC3fSAtnju/view?usp=sharing) \| [shell](../tools/shells/resnet50_baseline_tusimple.sh) |
| Baseline | ResNet101 | simple | 95.19% | 0.0619 | 0.0620 | [model](https://drive.google.com/file/d/1mELtKB3e8ntOmPovhnMphXWKf_bv83ef/view?usp=sharing) \| [shell](../tools/shells/resnet101_baseline_tusimple.sh) |
| Baseline | ERFNet | simple | 96.04% | 0.0591 | 0.0365 | [model](https://drive.google.com/file/d/1rLWDP_dkIQ7sBsCEzJi8T7ET1EPghhJJ/view?usp=sharing) \| [shell](../tools/shells/erfnet_baseline_tusimple.sh) |
| Baseline | ERFNet | strong | 94.37% | 0.0846 | 0.0770 | [model](https://drive.google.com/file/d/1LPmxT8rnyZL2M08lSLrlvrM0H_hMrFvq/view?usp=sharing) \| [shell](../tools/shells/erfnet_baseline-aug_tusimple.sh) |
| Baseline | ENet | simple | 95.61% | 0.0655 | 0.0503 | [model](https://drive.google.com/file/d/1CNSox62ghs0ArDVJb9mTZ1NVvqSkUNYC/view?usp=sharing) \| [shell](../tools/shells/enet_baseline_tusimple.sh) |
| SCNN | VGG16 | simple | 95.17% | 0.0637 | 0.0622 | [model](https://drive.google.com/file/d/1Fd46-f_8q-fGcJEI_PhPyh7aBY1uqbIw/view?usp=sharing) \| [shell](../tools/shells/vgg16_scnn_tusimple.sh) |
| SCNN | ResNet18 | simple | 94.77% | 0.0753 | 0.0737 | [model](https://drive.google.com/file/d/1cHp9gG2NgtC1iSp2LZMPF_UKiCb-fVkn/view?usp=sharing) \| [shell](../tools/shells/resnet18_scnn_tusimple.sh) |
| SCNN | ResNet34 | simple | 95.25% | 0.0627 | 0.0634 | [model](https://drive.google.com/file/d/1M0ROpEHV8DGJT4xWq2eURcbqMzpea1q7/view?usp=sharing) \| [shell](../tools/shells/resnet34_scnn_tusimple.sh) |
| SCNN | ResNet34 | strong | 93.42% | 0.0868 | 0.0998 | [model](https://drive.google.com/file/d/1t-cmUjBbLzSjODMvcpwoRPJKnUxLfWKk/view?usp=sharing) \| [shell](../tools/shells/resnet34_scnn-aug_tusimple.sh) |
| SCNN | ResNet50 | simple | 95.56% | 0.0561 | 0.0556 | [model](https://drive.google.com/file/d/1YK-PzdE9q8zn48isiBxwaZEdRsFw_oHe/view?usp=sharing) \| [shell](../tools/shells/resnet50_scnn_tusimple.sh) |
| SCNN | ResNet101 | simple | 95.69% | 0.0519 | 0.0504 | [model](https://drive.google.com/file/d/13qk5rIHqhDlwylZP9S-8fN53DexPTBQy/view?usp=sharing) \| [shell](../tools/shells/resnet101_scnn_tusimple.sh) |
| SCNN | ERFNet | simple | 96.29% | 0.0470 | 0.0318 | [model](https://drive.google.com/file/d/1rzE2fZ5mQswMIm6ICK1lWH-rsQyjRbxL/view?usp=sharing) \| [shell](../tools/shells/erfnet_scnn_tusimple.sh) |
| SCNN | ERFNet | strong | 95.26% | 0.0625 | 0.0512 | [model](https://drive.google.com/file/d/14XJ-W_wIOndjkhtPiUghwy0cLXrnl0PS/view?usp=sharing) \| [shell](../tools/shells/erfnet_scnn-aug_tusimple.sh) |
| LSTR | ResNet18s | strong | 95.06% | 0.0486 | 0.0418 | [model](https://drive.google.com/file/d/1z1ikrcgboyLFO3ysJUIf8qlBv7zEUvjK/view?usp=sharing) \| [shell](../tools/shells/resnet18s_lstr-aug_tusimple.sh) |
| LSTR | ResNet18s | simple | 92.40% | 0.1289 | 0.1127 | [model](https://drive.google.com/file/d/1iHArGHnOlSbS01RPFlLYI1mPJSX7o4sR/view?usp=sharing) \| [shell](../tools/shells/resnet18s_lstr_tusimple.sh) |

### CULane detailed performance (best):

| method | backbone | data<br>augmentation | normal | crowded | night | no line | shadow | arrow | dazzle<br>light | curve | crossroad | total | |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Baseline | VGG16 | simple | 85.51 | 64.05 | 61.14 | 35.96 | 59.76 | 78.43 | 53.25 | 62.16 | 2224 | 66.09 | [model](https://drive.google.com/file/d/1wVz1a7S1e5Dgy7ERk7E8dqQ8gyK-dWLG/view?usp=sharing) \| [shell](../tools/shells/vgg16_baseline_culane.sh) |
| Baseline | ResNet18 | simple | 85.45 | 62.63 | 61.04 | 33.88 | 51.72 | 78.15 | 53.05 | 59.70 | 1915 | 65.30 | [model](https://drive.google.com/file/d/1wkaTp8v1ceXrd6AjRccqpNxxxkd_qg1U/view?usp=sharing) \| [shell](../tools/shells/resnet18_baseline_culane.sh) |
| Baseline | ResNet34 | simple | 89.46 | 66.66 | 65.38 | 40.43 | 62.17 | 83.18 | 58.51 | 63.00 | 1713 | 69.92 | [model](https://drive.google.com/file/d/16VIJcd3wDOjFjg3UCVekUPcAb_F1K604/view?usp=sharing) \| [shell](../tools/shells/resnet34_baseline_culane.sh) |
| Baseline | ResNet50 | simple | 88.15 | 65.73 | 63.74 | 37.96 | 62.59 | 81.68 | 59.47 | 64.01 | 2046 | 68.48 | [model](https://drive.google.com/file/d/1DYVeH9kdSPhEMA4fsJFdEiw8qOwvQBl8/view?usp=sharing) \| [shell](../tools/shells/resnet50_baseline_culane.sh) |
| Baseline | ResNet101 | simple | 90.11 | 67.89 | 67.01 | 43.10 | 70.56 | 85.09 | 61.77 | 65.47 | 1883 | 71.37 | [model](https://drive.google.com/file/d/1iubFjWetsKE2VI4BEIWLDd80gB7IQUaP/view?usp=sharing) \| [shell](../tools/shells/resnet101_baseline_culane.sh) |
| Baseline | ERFNet | simple | 91.48 | 71.27 | 68.09 | 46.76 | 74.47 | 86.09 | 64.18 | 66.89 | 2102 | 73.49 | [model](https://drive.google.com/file/d/16-Q_jZYc9IIKUEHhClSTwZI4ClMeVvQS/view?usp=sharing) \| [shell](../tools/shells/erfnet_baseline_culane.sh) |
| Baseline | ENet | simple | 89.26 | 68.15 | 62.99 | 42.43 | 68.59 | 83.10 | 58.49 | 63.23 | 2464 | 69.90 | [model](https://drive.google.com/file/d/1DNgOpAVq87GIPUeAdMP6fnS4LhUmqyRB/view?usp=sharing) \| [shell](../tools/shells/enet_baseline_culane.sh) |
| SCNN | VGG16 | simple | 92.02 | 72.31 | 69.13 | 46.01 | 76.37 | 87.71 | 64.68 | 68.96 | 1924 | 74.29 | [model](https://drive.google.com/file/d/1vm8B1SSH0nlAIbz3aEGC1kqWP4YdFb3A/view?usp=sharing) \| [shell](../tools/shells/vgg16_scnn_culane.sh) |
| SCNN | ResNet18 | simple | 90.98 | 70.17 | 66.54 | 43.12 | 66.31 | 85.62 | 62.20 | 65.58 | 1808 | 72.19 | [model](https://drive.google.com/file/d/1i08KOS3b0hOTuzn866j4oUWtw3TYcndn/view?usp=sharing) \| [shell](../tools/shells/resnet18_scnn_culane.sh) |
| SCNN | ResNet34 | simple | 91.06 | 70.41 | 67.75 | 44.64 | 68.98 | 86.50 | 61.57 | 65.75 | 2017 | 72.70 | [model](https://drive.google.com/file/d/1JyPJQv8gpFZbr1sh7zRRiekUuDR4Aea8/view?usp=sharing) \| [shell](../tools/shells/resnet34_scnn_culane.sh) |
| SCNN | ResNet50 | simple | 91.38 | 70.60 | 67.62 | 45.02 | 71.24 | 86.90 | 66.03 | 66.17 | 1958 | 73.03 | [model](https://drive.google.com/file/d/1DxqUONEpT47RvJlCg7kDWIsdkYH0Fv8E/view?usp=sharing) \| [shell](../tools/shells/resnet50_scnn_culane.sh) |
| SCNN | ResNet101 | simple | 91.10 | 71.43 | 68.53 | 46.39 | 72.61 | 86.87 | 61.95 | 67.01 | 1720 | 73.58 | [model](https://drive.google.com/file/d/11O4ZDvNqQsKodnl9kJar6Mx1UKnB70L9/view?usp=sharing) \| [shell](../tools/shells/resnet101_scnn_culane.sh) |
| SCNN | ERFNet | simple | 91.82 | 72.13 | 69.49 | 46.68 | 70.59 | 87.40 | 64.18 | 68.30 | 2236 | 74.03 | [model](https://drive.google.com/file/d/1YOAuIJqh0M1RsPN5zISY7kTx9xt29IS3/view?usp=sharing) \| [shell](../tools/shells/erfnet_scnn_culane.sh) |
| LSTR | ResNet18s-2X | simple | 55.95 | 38.66 | 22.52 | 25.34 | 25.87 | 51.90 | 40.34 | 29.74 | 2128 | 39.17 | [model](https://drive.google.com/file/d/1vdYwM0xDcQLjMAibjmls8hX-IsUe0xcq/view?usp=sharing) \| [shell](../tools/shells/resnet18s_lstr_culane.sh) |
| LSTR | ResNet18s-2X | strong | 86.89 | 66.20 | 58.76 | 40.63 | 61.10 | 79.08 | 56.86 | 56.99 | 2455 | 67.21 | [model](https://drive.google.com/file/d/11Tv_nowlWmQtTYQfhGsziDIzb20kPo8o/view?usp=sharing) \| [shell](../tools/shells/resnet18s_lstr-aug_culane.sh) |
| LSTR | ResNet34 | strong | 89.68 | 68.85 | 65.59 | 45.50 | 67.71 | 84.76 | 64.12 | 64.08 | 1776 | 71.51 | [model](https://drive.google.com/file/d/1KfmXubuAtUoE9MO8iViMyB_3XhTxWnwH/view?usp=sharing) \| [shell](../tools/shells/resnet34_lstr-aug_culane.sh) |

### LLAMAS detailed performance (best):

| method | backbone | data<br>augmentation | F1 | TP | FP | FN | Precision | Recall | val / test | | 
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | 
| Baseline | VGG16 | simple | 95.11 | 70263 | 3460 | 3772 | 95.31 | 94.91 | val | [model](https://drive.google.com/file/d/1k2b7iRw3_YMJDMUsKVjtdOXRdZsXNXnO/view?usp=sharing) \| [shell](../tools/shells/vgg16_baseline_llamas.sh) |
| SCNN | VGG16 | simple | 96.42 | 71274 | 2526 | 2761 | 96.27 | 96.42 | val | [model](https://drive.google.com/file/d/1qE-euGGMZTxcHED_VDUW6eR-SKPkUDD-/view?usp=sharing) \| [shell](../tools/shells/vgg16_scnn_llamas.sh) |
| Baseline | ResNet34 | simple | 95.91 | 70841 | 2847 | 3194 | 96.14 | 95.69 | val | [model](https://drive.google.com/file/d/1YXNgwhQqwxoMkHDbRqAdnuAXIe1IdLSm/view?usp=sharing) \| [shell](../tools/shells/resnet34_baseline_llamas.sh) |
| SCNN | ResNet34 | simple | 96.19 | 71109 | 2705 | 2926 | 96.34 | 96.05 | val | [model](https://drive.google.com/file/d/1-vribu32iXViBqYumApmKQK4mjQd6BEp/view?usp=sharing) \| [shell](../tools/shells/resnet34_scnn_llamas.sh) |
| Baseline | ERFNet | simple | 96.13 | 71136 | 2830 | 2899 | 96.17 | 96.08 | val | [model](https://drive.google.com/file/d/15oNU4iffIPYTuKSCH2j0boRkYmm3-uSh/view?usp=sharing) \| [shell](../tools/shells/erfnet_baseline_llamas.sh) |
| SCNN | ERFNet | simple | 95.94 | 71036 | 3019 | 2999 | 95.92 | 95.95 | val | [model](https://drive.google.com/file/d/1oTdmP_tsguqa1-6bBIikT4gPThUVdXCr/view?usp=sharing) \| [shell](../tools/shells/erfnet_scnn_llamas.sh) |



## Semantic segmentation performance:

| model | resolution | mixed precision? | dataset | average | best | training time<br>*(2080 Ti)* | best model link |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| FCN | 321 x 321 | *yes* | PASCAL VOC 2012 | 70.72 | 70.83 | 3.3h | [model](https://drive.google.com/file/d/1SIIpApBdL0wXanlLeLWBSJJmX3AYLBnf/view?usp=sharing) \| [shell](../tools/shells/fcn_pascalvoc_321x321.sh) |
| FCN | 321 x 321 | *no* | PASCAL VOC 2012 | 70.91 | 71.55 | 6.3h | [model](https://drive.google.com/file/d/1ZunsGFjXxSIwR8Blckyk-Ils6IdhSqV1/view?usp=sharing) \| [shell](../tools/shells/fcn_pascalvoc_321x321_fp32.sh) |
| DeeplabV2 | 321 x 321 | *yes* | PASCAL VOC 2012 | 74.59 | 74.74 | 3.3h | [model](https://drive.google.com/file/d/1UGR4u1qvJcczLfcgmSHoVd0CGqHMfLoU/view?usp=sharing) \| [shell](../tools/shells/deeplabv2_pascalvoc_321x321.sh) |
| DeeplabV3 | 321 x 321 | *yes* | PASCAL VOC 2012 | 78.11 | 78.17 | 7h | [model](https://drive.google.com/file/d/1iYN73iqDD74HPZFGorARb6T2w7KkhbPM/view?usp=sharing) \| [shell](../tools/shells/deeplabv3_pascalvoc_321x321.sh) |
| FCN | 256 x 512 | *yes* | Cityscapes | 68.05 | 68.20 | 2.2h | [model](https://drive.google.com/file/d/1zT-lBElfkD1Sratu4WYiTCRU9PF16lLj/view?usp=sharing) \| [shell](../tools/shells/fcn_cityscapes_256x512.sh) |
| DeeplabV2 | 256 x 512 | *yes* | Cityscapes | 68.65 | 68.90 | 2.2h | [model](https://drive.google.com/file/d/16SR6EEdsuOtU6xyu7BsP-GQ16-y3OfGe/view?usp=sharing) \| [shell](../tools/shells/deeplabv2_cityscapes_256x512.sh) |
| DeeplabV3 | 256 x 512 | *yes* | Cityscapes | 69.87 | 70.37 | 4.5h | [model](https://drive.google.com/file/d/1HUR09zcPpjD5Q3LAm4p5t7e9gl1ZkpqU/view?usp=sharing) \| [shell](../tools/shells/deeplabv3_cityscapes_256x512.sh) |
| DeeplabV2 | 256 x 512 | *no* | Cityscapes | 68.45 | 68.89 | 4h | [model](https://drive.google.com/file/d/1fbxsPGu31plfgyQ0N0eiqk659F9osbRm/view?usp=sharing) \| [shell](../tools/shells/deeplabv2_cityscapes_256x512_fp32.sh) |
| ERFNet | 512 x 1024 | *yes* | Cityscapes | 71.99 | 72.47 | 5h | [model](https://drive.google.com/file/d/1uzBSboKD-Xt0K6VHd2aF561Cy13q9xRe/view?usp=sharing) \| [shell](../tools/shells/erfnet_cityscapes_512x1024.sh) |
| ENet | 512 x 1024 | *yes* | Cityscapes | 65.54 | 65.74 | 10.6h | [model](https://drive.google.com/file/d/1oK2mKCetOtY8KFaKLjs7-jOMkxZjbIQD/view?usp=sharing) \| [shell](../tools/shells/enet_cityscapes_512x1024.sh) |
| DeeplabV2 | 512 x 1024 | *yes* | Cityscapes | 71.78 | 72.12 | 9h | [model](https://drive.google.com/file/d/1MUG3PMMlFOtiX7G-TYCZhG_8D9aLqTPE/view?usp=sharing) \| [shell](../tools/shells/deeplabv2_cityscapes_512x1024.sh) |
| DeeplabV3 | 512 x 1024 | *yes* | Cityscapes | 74.64 | 74.67 | 20.1h | [model](https://drive.google.com/file/d/11xrX9AdNBdupSb8cOdGASiWrjYBXdQ48/view?usp=sharing) \| [shell](../tools/shells/deeplabv3_cityscapes_512x1024.sh) |
| DeeplabV2 | 512 x 1024 | *yes* | GTAV | 32.90 | 33.88 | 13.8h | [model](https://drive.google.com/file/d/1udHozZzwka9ktMxaV0tynL1HToy0H6sI/view?usp=sharing) \| [shell](../tools/shells/deeplabv2_gtav_512x1024.sh) |
| DeeplabV2 | 512 x 1024 | *yes* | SYNTHIA* | 33.89 | 34.86 | 10.4h | [model](https://drive.google.com/file/d/1M-CO46zjXbVo8pguISUEw3M6NoKHIN0l/view?usp=sharing) \| [shell](../tools/shells/deeplabv2_synthia_512x1024.sh) |

*All performance is measured with ImageNet pre-training and reported as 3 times average/best mIoU (%) on val set.*

*\* mIoU-16.*
