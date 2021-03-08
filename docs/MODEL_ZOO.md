# Welcome to pytorch-auto-drive model zoo

## Lane detection performance:

| method | backbone | resolution | mixed precision? | dataset | metric | average | best | training time |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Baseline | VGG16 | 360 x 640 | *yes* | Tusimple | Accuracy | 93.39% | 93.57% | 1.5h |
| Baseline | ResNet18 | 360 x 640 | *yes* | TuSimple | Accuracy | 93.80% | 93.98% | 0.7h |
| Baseline | ResNet34 | 360 x 640 | *yes* | TuSimple | Accuracy | 94.94% | 94.99% | 1.1h |
| Baseline | ResNet50 | 360 x 640 | *yes* | TuSimple | Accuracy | 94.67% | 94.71% | 1.5h |
| Baseline | ResNet101 | 360 x 640 | *yes* | TuSimple | Accuracy | 94.77% | 94.83% | 2.6h |
| Baseline | ERFNet | 360 x 640 | *yes* | TuSimple | Accuracy | 95.15% | 95.24% | 0.8h |
| SCNN | VGG16 | 360 x 640 | *yes* | Tusimple | Accuracy | 94.46% | 94.65% | 2h |
| SCNN | ResNet18 | 360 x 640 | *yes* | TuSimple | Accuracy | 94.21% | 94.30% | 1.2h |
| SCNN | ResNet34 | 360 x 640 | *yes* | TuSimple | Accuracy | 94.66% | 94.76% | 1.6h |
| SCNN | ResNet50 | 360 x 640 | *yes* | TuSimple | Accuracy | 94.93% | 95.01% | 2.4h |
| SCNN | ResNet101 | 360 x 640 | *yes* | TuSimple | Accuracy | 95.09% | 95.21% | 3.5h |
| SCNN | ERFNet | 360 x 640 | *yes* | TuSimple | Accuracy | 96.00% | 96.12% | 1.6h |
| Baseline | VGG16 | 288 x 800 | *yes* | CULane | F measure | 65.93 | 66.09 | 9.3h |
| Baseline | ResNet18 | 288 x 800 | *yes* | CULane | F measure | 65.19 | 65.30 | 5.3h |
| Baseline | ResNet34 | 288 x 800 | *yes* | CULane | F measure | 69.82 | 69.92 | 7.3h |
| Baseline | ERFNet | 288 x 800 | *yes* | CULane | F measure | 73.40 | 73.49 | 6h |
| SCNN | VGG16 | 288 x 800 | *yes* | CULane | F measure | 73.13 | 73.23 | 12.8h |
| SCNN | ResNet18 | 288 x 800 | *yes* | CULane | F measure | 71.94 | 72.19 | 8.0h |
| SCNN | ResNet34 | 288 x 800 | *yes* | CULane | F measure | 72.44 | 72.70 | 10.7h |
| SCNN | ERFNet | 288 x 800 | *yes* | CULane | F measure | 73.85 | 74.03 | 11.3h |

*All performance is measured with ImageNet pre-training and reported as 3 times average/best on test set.*

### Tusimple detailed performance (best):

| method | backbone | accuracy | FP | FN | |
| :---: | :---: | :---: | :---: | :---: | :---: |
| Baseline | VGG16 | 93.57% | 0.0992 | 0.1058 | [model](https://drive.google.com/file/d/1ChK0hApqLU0xUiEm4Wul-gNYQDQka151/view?usp=sharing) |
| Baseline | ResNet18 | 93.98% | 0.0874 | 0.0921 | [model](https://drive.google.com/file/d/17VKnwsN4WMbpnD4DgaaerppjXybqn-LG/view?usp=sharing) |
| Baseline | ResNet34 | 94.99% | 0.0615 | 0.0638 | [model](https://drive.google.com/file/d/1ch5YCNAQPhkPR2PRBNn07kE_t8kicLpq/view?usp=sharing) |
| Baseline | ResNet50 | 94.71% | 0.0644 | 0.0695 | [model](https://drive.google.com/file/d/10KBMVGc63kPvqL_2deaLfTfC3fSAtnju/view?usp=sharing) |
| Baseline | ResNet101 | 94.83% | 0.0612 | 0.0677 | [model](https://drive.google.com/file/d/1sFJna_oJ6dfd9AMiAEbragpLSvpduGff/view?usp=sharing) |
| Baseline | ERFNet | 95.24% | 0.0569 | 0.0457 | [model](https://drive.google.com/file/d/12n_ck3Ir86j3VOhIn0hT96Ru4n8nhP5G/view?usp=sharing) |
| SCNN | VGG16 | 94.65% | 0.0627 | 0.0675 | [model](https://drive.google.com/file/d/1Fd46-f_8q-fGcJEI_PhPyh7aBY1uqbIw/view?usp=sharing) |
| SCNN | ResNet18 | 94.30% | 0.0736 | 0.0799 | [model](https://drive.google.com/file/d/1cmP73GKD_9R9ka0sJdY8V0z04bE5jkaZ/view?usp=sharing) |
| SCNN | ResNet34 | 94.76% | 0.0671 | 0.0694 | [model](https://drive.google.com/file/d/1LHlnPsIsr4RCJar4UKBVfikS41P9e3Em/view?usp=sharing) |
| SCNN | ResNet50 | 95.01% | 0.0550 | 0.0611 | [model](https://drive.google.com/file/d/1YK-PzdE9q8zn48isiBxwaZEdRsFw_oHe/view?usp=sharing) |
| SCNN | ResNet101 | 95.21% | 0.0511 | 0.0552 | [model](https://drive.google.com/file/d/13qk5rIHqhDlwylZP9S-8fN53DexPTBQy/view?usp=sharing) |
| SCNN | ERFNet | 96.12% | 0.0468 | 0.0335 | [model](https://drive.google.com/file/d/1rzE2fZ5mQswMIm6ICK1lWH-rsQyjRbxL/view?usp=sharing) |

### CULane detailed performance (best):

| method | backbone | normal | crowded | night | no line | shadow | arrow | dazzle<br>light | curve | crossroad | total | |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Baseline | VGG16 | 85.51 | 64.05 | 61.14 | 35.96 | 59.76 | 78.43 | 53.25 | 62.16 | 2224 | 66.09 | |
| Baseline | ResNet18 | 85.45 | 62.63 | 61.04 | 33.88 | 51.72 | 78.15 | 53.05 | 59.70 | 1915 | 65.30 | [model](https://drive.google.com/file/d/1wkaTp8v1ceXrd6AjRccqpNxxxkd_qg1U/view?usp=sharing) |
| Baseline | ResNet34 | 89.46 | 66.66 | 65.38 | 40.43 | 62.17 | 83.18 | 58.51 | 63.00 | 1713 | 69.92 | [model](https://drive.google.com/file/d/16VIJcd3wDOjFjg3UCVekUPcAb_F1K604/view?usp=sharing) |
| Baseline | ERFNet | 91.48 | 71.27 | 68.09 | 46.76 | 74.47 | 86.09 | 64.18 | 66.89 | 2102 | 73.49 | [model](https://drive.google.com/file/d/16-Q_jZYc9IIKUEHhClSTwZI4ClMeVvQS/view?usp=sharing) |
| SCNN | VGG16 | 91.17 | 71.56 | 67.83 | 45.59 | 69.38 | 86.56 | 65.80 | 66.58 | 1809 | 73.23 | |
| SCNN | ResNet18 | 90.98 | 70.17 | 66.54 | 43.12 | 66.31 | 85.62 | 62.20 | 65.58 | 1808 | 72.19 | [model](https://drive.google.com/file/d/1i08KOS3b0hOTuzn866j4oUWtw3TYcndn/view?usp=sharing) |
| SCNN | ResNet34 | 91.06 | 70.41 | 67.75 | 44.64 | 68.98 | 86.50 | 61.57 | 65.75 | 2017 | 72.70 | [model](https://drive.google.com/file/d/1JyPJQv8gpFZbr1sh7zRRiekUuDR4Aea8/view?usp=sharing) |
| SCNN | ERFNet | 91.82 | 72.13 | 69.49 | 46.68 | 70.59 | 87.40 | 64.18 | 68.30 | 2236 | 74.03 | [model](https://drive.google.com/file/d/1YOAuIJqh0M1RsPN5zISY7kTx9xt29IS3/view?usp=sharing) |

## Semantic segmentation performance:

| model | resolution | mixed precision? | dataset | average | best | training time | best model link |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| FCN | 321 x 321 | *yes* | PASCAL VOC 2012 | 70.72 | 70.83 | 3.3h | [model](https://drive.google.com/file/d/1SIIpApBdL0wXanlLeLWBSJJmX3AYLBnf/view?usp=sharing) |
| FCN | 321 x 321 | *no* | PASCAL VOC 2012 | 70.91 | 71.55 | 6.3h | [model](https://drive.google.com/file/d/1ZunsGFjXxSIwR8Blckyk-Ils6IdhSqV1/view?usp=sharing) |
| DeeplabV2 | 321 x 321 | *yes* | PASCAL VOC 2012 | 74.59 | 74.74 | 3.3h | [model](https://drive.google.com/file/d/1UGR4u1qvJcczLfcgmSHoVd0CGqHMfLoU/view?usp=sharing) |
| DeeplabV3 | 321 x 321 | *yes* | PASCAL VOC 2012 | 78.11 | 78.17 | 7h | [model](https://drive.google.com/file/d/1iYN73iqDD74HPZFGorARb6T2w7KkhbPM/view?usp=sharing) |
| FCN | 256 x 512 | *yes* | Cityscapes | 68.05 | 68.20 | 2.2h | [model](https://drive.google.com/file/d/1zT-lBElfkD1Sratu4WYiTCRU9PF16lLj/view?usp=sharing) |
| DeeplabV2 | 256 x 512 | *yes* | Cityscapes | 68.65 | 68.90 | 2.2h | [model](https://drive.google.com/file/d/16SR6EEdsuOtU6xyu7BsP-GQ16-y3OfGe/view?usp=sharing) |
| DeeplabV3 | 256 x 512 | *yes* | Cityscapes | 69.87 | 70.37 | 4.5h | [model](https://drive.google.com/file/d/1HUR09zcPpjD5Q3LAm4p5t7e9gl1ZkpqU/view?usp=sharing) |
| DeeplabV2 | 256 x 512 | *no* | Cityscapes | 68.45 | 68.89 | 4h | [model](https://drive.google.com/file/d/1fbxsPGu31plfgyQ0N0eiqk659F9osbRm/view?usp=sharing) |
| ERFNet | 512 x 1024 | *yes* | Cityscapes | 71.99 | 72.47 | 5h | [model](https://drive.google.com/file/d/1uzBSboKD-Xt0K6VHd2aF561Cy13q9xRe/view?usp=sharing) |
| ENet | 512 x 1024 | *yes* | Cityscapes | 65.54 | 65.74 | 10.6h | [model](https://drive.google.com/file/d/1oK2mKCetOtY8KFaKLjs7-jOMkxZjbIQD/view?usp=sharing) |
| DeeplabV2 | 512 x 1024 | *yes* | Cityscapes | 71.78 | 72.12 | 9h | [model](https://drive.google.com/file/d/1MUG3PMMlFOtiX7G-TYCZhG_8D9aLqTPE/view?usp=sharing) |
| DeeplabV2 | 512 x 1024 | *yes* | GTAV | 32.90 | 33.88 | 13.8h | [model](https://drive.google.com/file/d/1udHozZzwka9ktMxaV0tynL1HToy0H6sI/view?usp=sharing) |
| DeeplabV2 | 512 x 1024 | *yes* | SYNTHIA* | 33.89 | 34.86 | 10.4h | [model](https://drive.google.com/file/d/1M-CO46zjXbVo8pguISUEw3M6NoKHIN0l/view?usp=sharing) |

*All performance is measured with ImageNet pre-training and reported as 3 times average/best mIoU (%) on val set.*

*\*SYHTHIA performance is mIoU-16.*
