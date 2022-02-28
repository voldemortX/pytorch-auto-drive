# Welcome to PytorchAutoDrive model zoo

## Lane detection performance

**Data Augmentation levels:**

 - **level 0**: only small rotation and resize
 - **level 1a**: the LSTR augmentations
 - **level 1b**: the BezierLaneNet augmentations

| method | backbone | data<br>augmentation | resolution | mixed precision? | dataset | metric | average | best | training time <br> *(2080 Ti)* |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Baseline | VGG16 | level 0 | 360 x 640 | *yes* | TuSimple | Accuracy | 93.79% | 93.94% | 1.5h |
| Baseline | ResNet18 | level 0 | 360 x 640 | *yes* | TuSimple | Accuracy | 94.18% | 94.25% | 0.7h |
| Baseline | ResNet34 | level 0 | 360 x 640 | *yes* | TuSimple | Accuracy | 95.23% | 95.31% | 1.1h |
| Baseline | ResNet34 | level 1a | 360 x 640 | *no* | TuSimple | Accuracy | 92.14% | 92.68% | 1.2h* |
| Baseline | ResNet50 | level 0 | 360 x 640 | *yes* | TuSimple | Accuracy | 95.07% | 95.12% | 1.5h |
| Baseline | ResNet101 | level 0 | 360 x 640 | *yes* | TuSimple | Accuracy | 95.15% | 95.19% | 2.6h |
| Baseline | ERFNet | level 0 | 360 x 640 | *yes* | TuSimple | Accuracy | 96.02% | 96.04% | 0.8h |
| Baseline | ERFNet | level 1a | 360 x 640 | *no* | TuSimple | Accuracy | 94.21% | 94.37% | 0.9h* |
| Baseline | ENet<sup>#</sup> | level 0 | 360 x 640 | *yes* | TuSimple | Accuracy | 95.55% | 95.61% | 1h<sup>+</sup> |
| Baseline | MobileNetV2 | level 0 | 360 x 640 | *yes* | TuSimple | Accuracy | 93.98% | 94.07% | 0.5h |
| Baseline | MobileNetV3-Large | level 0 | 360 x 640 | *yes* | TuSimple | Accuracy | 92.09% | 92.18% | 0.5h |
| SCNN | VGG16 | level 0 | 360 x 640 | *yes* | TuSimple | Accuracy | 95.01% | 95.17% | 2h |
| SCNN | ResNet18 | level 0 | 360 x 640 | *yes* | TuSimple | Accuracy | 94.69% | 94.77% | 1.2h |
| SCNN | ResNet34 | level 0 | 360 x 640 | *yes* | TuSimple | Accuracy | 95.19% | 95.25% | 1.6h |
| SCNN | ResNet34 | level 1a | 360 x 640 | *no* | TuSimple | Accuracy | 92.62% | 93.42% | 1.7h* |
| SCNN | ResNet50 | level 0 | 360 x 640 | *yes* | TuSimple | Accuracy | 95.43% | 95.56% | 2.4h |
| SCNN | ResNet101 | level 0 | 360 x 640 | *yes* | TuSimple | Accuracy | 95.56% | 95.69% | 3.5h |
| SCNN | ERFNet | level 0 | 360 x 640 | *yes* | TuSimple | Accuracy | 96.18% | 96.29% | 1.6h |
| SCNN | ERFNet | level 1a | 360 x 640 | *no* | TuSimple | Accuracy | 95.00% | 95.26% | 1.7h* |
| RESA | ResNet18 | level 0 | 360 x 640 | *no* | TuSimple | Accuracy | 94.64% | 95.24% | 1.2h* |
| RESA | ResNet34 | level 0 | 360 x 640 | *no* | TuSimple | Accuracy | 94.84% | 95.15% | 1.6h* |
| RESA | ResNet50 | level 0 | 360 x 640 | *no* | TuSimple | Accuracy | 95.34% | 95.50% | 2.4h* |
| RESA | ResNet101 | level 0 | 360 x 640 | *no* | TuSimple | Accuracy | 95.24% | 95.56% | 3.5h* |
| RESA | MobileNetV2 | level 0 | 360 x 640 | *yes* | TuSimple | Accuracy | 94.61% | 95.21% | 0.7h |
| RESA | MobileNetV3-Large | level 0 | 360 x 640 | *yes* | TuSimple | Accuracy | 94.56% | 94.99% | 0.7h |
| LSTR | ResNet18s<sup>#</sup> | level 0 | 360 x 640 | *no* | TuSimple | Accuracy | 91.91% | 92.40% | 14.2h |
| LSTR | ResNet18s<sup>#</sup> | level 1a | 360 x 640 | *no* | TuSimple | Accuracy | 94.91% | 95.06% | 15.5h |
| BezierLaneNet | ResNet18 | level 1b | 360 x 640 | *no* | TuSimple | Accuracy | 95.01% | 95.41% | 5.5h |
| BezierLaneNet | ResNet34 | level 1b | 360 x 640 | *no* | TuSimple | Accuracy | 95.17% | 95.65% | 6.5h |
| Baseline | VGG16 | level 0 | 288 x 800 | *yes* | CULane | F measure | 65.93 | 66.09 | 9.3h |
| Baseline | ResNet18 | level 0 | 288 x 800 | *yes* | CULane | F measure | 65.19 | 65.30 | 5.3h |
| Baseline | ResNet34 | level 0 | 288 x 800 | *yes* | CULane | F measure | 69.82 | 69.92 | 7.3h |
| Baseline | ResNet50 | level 0 | 288 x 800 | *yes* | CULane | F measure | 68.31 | 68.48 | 12.4h |
| Baseline | ResNet101 | level 0 | 288 x 800 | *yes* | CULane | F measure | 71.29 | 71.37 | 20.0h |
| Baseline | ERFNet | level 0 | 288 x 800 | *yes* | CULane | F measure | 73.40 | 73.49 | 6h |
| Baseline | ENet<sup>#</sup> | level 0 | 288 x 800 | *yes* | CULane | F measure | 69.39 | 69.90 | 6.4h<sup>+</sup> |
| Baseline | MobileNetV2 | level 0 | 288 x 800 | *yes* | CULane | F measure | 67.34 | 67.41 | 3.0h |
| Baseline | MobileNetV3-Large | level 0 | 288 x 800 | *yes* | CULane | F measure | 68.27 | 68.42 | 3.0h |
| Baseline | RepVGG-A0| level 0 | 288 x 800 | *yes* | CULane | F measure | 70.22 | 70.56 | 3.3h** |
| Baseline | RepVGG-A1 | level 0 | 288 x 800 | *yes* | CULane | F measure | 70.73 | 70.85 | 4.1h** |
| Baseline | RepVGG-B0 | level 0 | 288 x 800 | *yes* | CULane | F measure | 71.77 | 71.81 | 6.2h** |
| Baseline | RepVGG-B1g2 | level 0 | 288 x 800 | *yes* | CULane | F measure | 72.08 | 72.20 | 10.0h** |
| Baseline | RepVGG-B2 | level 0 | 288 x 800 | *yes* | CULane | F measure | 72.24 | 72.33 | 13.2h** |
| Baseline | Swin-Tiny | level 0 | 288 x 800 | *yes* | CULane | F measure | 69.75 | 69.90 | 12.1h** |
| SCNN | VGG16 | level 0 | 288 x 800 | *yes* | CULane | F measure | 74.02 | 74.29 | 12.8h |
| SCNN | ResNet18 | level 0 | 288 x 800 | *yes* | CULane | F measure | 71.94 | 72.19 | 8.0h |
| SCNN | ResNet34 | level 0 | 288 x 800 | *yes* | CULane | F measure | 72.44 | 72.70 | 10.7h |
| SCNN | ResNet50 | level 0 | 288 x 800 | *yes* | CULane | F measure | 72.95 | 73.03 | 17.9h |
| SCNN | ResNet101 | level 0 | 288 x 800 | *yes* | CULane | F measure | 73.29 | 73.58 | 25.7h |
| SCNN | ERFNet | level 0 | 288 x 800 | *yes* | CULane | F measure | 73.85 | 74.03 | 11.3h |
| SCNN | RepVGG-A1 | level 0 | 288 x 800 | *yes* | CULane | F measure | 72.88 | 72.89 | 5.7h** |
| RESA | ResNet18 | level 0 | 288 x 800 | *no* | CULane | F measure | 72.76 | 72.90 | 8.0h* |
| RESA | ResNet34 | level 0 | 288 x 800 | *no* | CULane | F measure | 73.29 | 73.66 | 10.7h* |
| RESA | ResNet50 | level 0 | 288 x 800 | *no* | CULane | F measure | 73.99 | 74.19 | 17.9h* |
| RESA | ResNet101 | level 0 | 288 x 800 | *no* | CULane | F measure | 73.96 | 74.04 | 25.7h* |
| RESA | MobileNetV2 | level 0 | 288 x 800 | *yes* | CULane | F measure | 72.28 | 72.36 | 4.6h |
| RESA | MobileNetV3-Large | level 0 | 288 x 800 | *yes* | CULane | F measure | 70.23 | 70.61 | 4.6h |
| LSTR | ResNet18s-2X<sup>#</sup> | level 0 | 288 x 800 | *no* | CULane | F measure | 36.27 | 39.77 | 28.5h* |
| LSTR | ResNet18s-2X<sup>#</sup> | level 1a | 288 x 800 | *no* | CULane | F measure | 68.35 | 68.72 | 31.5h* |
| LSTR | ResNet34 | level 1a | 288 x 800 | *no* | CULane | F measure | 72.17 | 72.48 | 45.0h* |
| BezierLaneNet | ResNet18 | level 1b | 288 x 800 | *yes* | CULane | F measure | 73.36 | 73.67 | 9.9h |
| BezierLaneNet | ResNet34 | level 1b | 288 x 800 | *yes* | CULane | F measure | 75.30 | 75.57 | 11.0h |
| Baseline | ERFNet | level 0 | 360 x 640 | *yes* | LLAMAS | F measure | 95.94 | 96.13 | 10.9h<sup>+</sup> |
| Baseline | VGG16 | level 0 | 360 x 640 | *yes* | LLAMAS | F measure | 95.05 | 95.11 | 9.3h |
| Baseline | ResNet34 | level 0 | 360 x 640 | *yes* | LLAMAS | F measure | 95.90 | 95.91 | 7.0h |
| SCNN | ERFNet | level 0 | 360 x 640 | *yes* | LLAMAS | F measure | 95.89 | 95.94 | 14.2h<sup>+</sup> |
| SCNN | VGG16 | level 0 | 360 x 640 | *yes* | LLAMAS | F measure | 96.39 | 96.42 | 12.5h |
| SCNN | ResNet34 | level 0 | 360 x 640 | *yes* | LLAMAS | F measure | 96.17 | 96.19 | 10.1h |
| BezierLaneNet | ResNet18 | level 1b | 360 x 640 | *yes* | LLAMAS | F measure | 95.42 | 95.52 | 5.5h |
| BezierLaneNet | ResNet34 | level 1b | 360 x 640 | *yes* | LLAMAS | F measure | 96.04 | 96.11 | 6.1h |

*All performance is measured with ImageNet pre-training and reported as 3 times average/best on test set.*

*The test set annotations of LLAMAS are not public, so we provide validation set result in this table.*

*<sup>+</sup> Measured on a single GTX 1080Ti.*

*<sup>#</sup> No pre-training.*

*\* Trained on a 1080 Ti cluster, with CUDA 9.0 PyTorch 1.3, training time is estimated as: single 2080 Ti, mixed precision.*

*\*\* Trained on two 2080ti.*  

### TuSimple detailed performance (best):

| method | backbone | data<br>augmentation | accuracy | FP | FN | |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Baseline | VGG16 | level 0 | 93.94% | 0.0998 | 0.1021 | [model](https://drive.google.com/file/d/1ChK0hApqLU0xUiEm4Wul-gNYQDQka151/view?usp=sharing) \| [shell](../tools/shells/vgg16_baseline_tusimple.sh) |
| Baseline | ResNet18 | level 0 | 94.25% | 0.0881 | 0.0894 | [model](https://drive.google.com/file/d/17VKnwsN4WMbpnD4DgaaerppjXybqn-LG/view?usp=sharing) \| [shell](../tools/shells/resnet18_baseline_tusimple.sh) |
| Baseline | ResNet34 | level 0 | 95.31% | 0.0640 | 0.0622 | [model](https://drive.google.com/file/d/1NAck0aQZK_wAHer4xB8xzegxDWk9EFtG/view?usp=sharing) \| [shell](../tools/shells/resnet34_baseline_tusimple.sh) |
| Baseline | ResNet34 | level 1a | 92.68% | 0.1073 | 0.1221 | [model](https://drive.google.com/file/d/1OhN2tWIep9ncKFf-_2RqUEaSJvPK60cn/view?usp=sharing) \| [shell](../tools/shells/resnet34_baseline-aug_tusimple.sh) |
| Baseline | ResNet50 | level 0 | 95.12% | 0.0649 | 0.0653 | [model](https://drive.google.com/file/d/10KBMVGc63kPvqL_2deaLfTfC3fSAtnju/view?usp=sharing) \| [shell](../tools/shells/resnet50_baseline_tusimple.sh) |
| Baseline | ResNet101 | level 0 | 95.19% | 0.0619 | 0.0620 | [model](https://drive.google.com/file/d/1mELtKB3e8ntOmPovhnMphXWKf_bv83ef/view?usp=sharing) \| [shell](../tools/shells/resnet101_baseline_tusimple.sh) |
| Baseline | ERFNet | level 0 | 96.04% | 0.0591 | 0.0365 | [model](https://drive.google.com/file/d/1rLWDP_dkIQ7sBsCEzJi8T7ET1EPghhJJ/view?usp=sharing) \| [shell](../tools/shells/erfnet_baseline_tusimple.sh) |
| Baseline | ERFNet | level 1a | 94.37% | 0.0846 | 0.0770 | [model](https://drive.google.com/file/d/1LPmxT8rnyZL2M08lSLrlvrM0H_hMrFvq/view?usp=sharing) \| [shell](../tools/shells/erfnet_baseline-aug_tusimple.sh) |
| Baseline | ENet | level 0 | 95.61% | 0.0655 | 0.0503 | [model](https://drive.google.com/file/d/1CNSox62ghs0ArDVJb9mTZ1NVvqSkUNYC/view?usp=sharing) \| [shell](../tools/shells/enet_baseline_tusimple.sh) |
| Baseline | MobileNetV2 | level 0 | 94.07% | 0.0792 | 0.0866 | [model](https://drive.google.com/file/d/1SUqt3BDXSMhAv68F9VIKncY0lDUg9My8/view?usp=sharing) \| [shell](../tools/shells/mobilenetv2_baseline_tusimple.sh) |
| Baseline | MobileNetV3-Large | level 0 | 92.18% | 0.1149 | 0.1322 | [model](https://drive.google.com/file/d/1I5SPlkmC8TnNeANoQGxzP3P1_iVxms3u/view?usp=sharing) \| [shell](../tools/shells/mobilenetv3-large_baseline_tusimple.sh) |
| SCNN | VGG16 | level 0 | 95.17% | 0.0637 | 0.0622 | [model](https://drive.google.com/file/d/1Fd46-f_8q-fGcJEI_PhPyh7aBY1uqbIw/view?usp=sharing) \| [shell](../tools/shells/vgg16_scnn_tusimple.sh) |
| SCNN | ResNet18 | level 0 | 94.77% | 0.0753 | 0.0737 | [model](https://drive.google.com/file/d/1cHp9gG2NgtC1iSp2LZMPF_UKiCb-fVkn/view?usp=sharing) \| [shell](../tools/shells/resnet18_scnn_tusimple.sh) |
| SCNN | ResNet34 | level 0 | 95.25% | 0.0627 | 0.0634 | [model](https://drive.google.com/file/d/1M0ROpEHV8DGJT4xWq2eURcbqMzpea1q7/view?usp=sharing) \| [shell](../tools/shells/resnet34_scnn_tusimple.sh) |
| SCNN | ResNet34 | level 1a | 93.42% | 0.0868 | 0.0998 | [model](https://drive.google.com/file/d/1t-cmUjBbLzSjODMvcpwoRPJKnUxLfWKk/view?usp=sharing) \| [shell](../tools/shells/resnet34_scnn-aug_tusimple.sh) |
| SCNN | ResNet50 | level 0 | 95.56% | 0.0561 | 0.0556 | [model](https://drive.google.com/file/d/1YK-PzdE9q8zn48isiBxwaZEdRsFw_oHe/view?usp=sharing) \| [shell](../tools/shells/resnet50_scnn_tusimple.sh) |
| SCNN | ResNet101 | level 0 | 95.69% | 0.0519 | 0.0504 | [model](https://drive.google.com/file/d/13qk5rIHqhDlwylZP9S-8fN53DexPTBQy/view?usp=sharing) \| [shell](../tools/shells/resnet101_scnn_tusimple.sh) |
| SCNN | ERFNet | level 0 | 96.29% | 0.0470 | 0.0318 | [model](https://drive.google.com/file/d/1rzE2fZ5mQswMIm6ICK1lWH-rsQyjRbxL/view?usp=sharing) \| [shell](../tools/shells/erfnet_scnn_tusimple.sh) |
| SCNN | ERFNet | level 1a | 95.26% | 0.0625 | 0.0512 | [model](https://drive.google.com/file/d/14XJ-W_wIOndjkhtPiUghwy0cLXrnl0PS/view?usp=sharing) \| [shell](../tools/shells/erfnet_scnn-aug_tusimple.sh) |
| RESA | ResNet18 | level 0 | 95.24% | 0.0685 | 0.0571 | [model](https://drive.google.com/file/d/1I3XpUB_I0SEkvE4sejIFyInOPZ5XsWyG/view?usp=sharing) \| [shell](../tools/shells/resnet18_resa_tusimple.sh) |
| RESA | ResNet34 | level 0 | 95.15% | 0.0690 | 0.0592 | [model](https://drive.google.com/file/d/1Spa1bCXoFyjCgOO-ordSPP5a3GMW6E0N/view?usp=sharing) \| [shell](../tools/shells/resnet34_resa_tusimple.sh) |
| RESA | ResNet50 | level 0 | 95.50% | 0.0550 | 0.0507 | [model](https://drive.google.com/file/d/1Mmb_4AFzSpZBcB7UxKr3Vlhwi27exqp0/view?usp=sharing) \| [shell](../tools/shells/resnet50_resa_tusimple.sh) |
| RESA | ResNet101 | level 0 | 95.56% | 0.0580 | 0.0513 | [model](https://drive.google.com/file/d/1i--g7uzt3dTeMlNnAXbBCSJ8RXVjIljR/view?usp=sharing) \| [shell](../tools/shells/resnet101_resa_tusimple.sh) |
| RESA | MobileNetV2 | level 0 | 95.21% | 0.0642 | 0.0552 | [model](https://drive.google.com/file/d/1XuQ-jak8qViNK9NXeeVmya2EQ-Z3XxZW/view?usp=sharing) \| [shell](../tools/shells/mobilenetv2_resa_tusimple.sh) |
| RESA | MobileNetV3-Large | level 0 | 94.99% | 0.0841 | 0.0597 | [model](https://drive.google.com/file/d/1ax7YTH6r8o9PIKSLT4fh7GVdaKgdujMO/view?usp=sharing) \| [shell](../tools/shells/mobilenetv3-large_resa_tusimple.sh) |
| LSTR | ResNet18s | level 1a | 95.06% | 0.0486 | 0.0418 | [model](https://drive.google.com/file/d/1z1ikrcgboyLFO3ysJUIf8qlBv7zEUvjK/view?usp=sharing) \| [shell](../tools/shells/resnet18s_lstr-aug_tusimple.sh) |
| LSTR | ResNet18s | level 0 | 92.40% | 0.1289 | 0.1127 | [model](https://drive.google.com/file/d/1iHArGHnOlSbS01RPFlLYI1mPJSX7o4sR/view?usp=sharing) \| [shell](../tools/shells/resnet18s_lstr_tusimple.sh) |
| BezierLaneNet | ResNet18 | level 1b | 95.41% | 0.0531 | 0.0458 | [model](https://drive.google.com/file/d/10qMdvPBnZP4P88EQXYZxsXZgj7sz6LvS/view?usp=sharing) \| [shell](../tools/shells/resnet18_bezierlanenet_tusimple-aug1b.sh) |
| BezierLaneNet | ResNet34 | level 1b | 95.65% | 0.0513 | 0.0386 | [model](https://drive.google.com/file/d/1FFn8j2BoUsyj8UbBcfeGWKvCQj9Qg-44/view?usp=sharing) \| [shell](../tools/shells/resnet34_bezierlanenet_tusimple-aug1b.sh) |

### CULane detailed performance (best):

| method | backbone | data<br>augmentation | normal | crowded | night | no line | shadow | arrow | dazzle<br>light | curve | crossroad | total | |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Baseline | VGG16 | level 0 | 85.51 | 64.05 | 61.14 | 35.96 | 59.76 | 78.43 | 53.25 | 62.16 | 2224 | 66.09 | [model](https://drive.google.com/file/d/1wVz1a7S1e5Dgy7ERk7E8dqQ8gyK-dWLG/view?usp=sharing) \| [shell](../tools/shells/vgg16_baseline_culane.sh) |
| Baseline | ResNet18 | level 0 | 85.45 | 62.63 | 61.04 | 33.88 | 51.72 | 78.15 | 53.05 | 59.70 | 1915 | 65.30 | [model](https://drive.google.com/file/d/1wkaTp8v1ceXrd6AjRccqpNxxxkd_qg1U/view?usp=sharing) \| [shell](../tools/shells/resnet18_baseline_culane.sh) |
| Baseline | ResNet34 | level 0 | 89.46 | 66.66 | 65.38 | 40.43 | 62.17 | 83.18 | 58.51 | 63.00 | 1713 | 69.92 | [model](https://drive.google.com/file/d/16VIJcd3wDOjFjg3UCVekUPcAb_F1K604/view?usp=sharing) \| [shell](../tools/shells/resnet34_baseline_culane.sh) |
| Baseline | ResNet50 | level 0 | 88.15 | 65.73 | 63.74 | 37.96 | 62.59 | 81.68 | 59.47 | 64.01 | 2046 | 68.48 | [model](https://drive.google.com/file/d/1DYVeH9kdSPhEMA4fsJFdEiw8qOwvQBl8/view?usp=sharing) \| [shell](../tools/shells/resnet50_baseline_culane.sh) |
| Baseline | ResNet101 | level 0 | 90.11 | 67.89 | 67.01 | 43.10 | 70.56 | 85.09 | 61.77 | 65.47 | 1883 | 71.37 | [model](https://drive.google.com/file/d/1iubFjWetsKE2VI4BEIWLDd80gB7IQUaP/view?usp=sharing) \| [shell](../tools/shells/resnet101_baseline_culane.sh) |
| Baseline | ERFNet | level 0 | 91.48 | 71.27 | 68.09 | 46.76 | 74.47 | 86.09 | 64.18 | 66.89 | 2102 | 73.49 | [model](https://drive.google.com/file/d/16-Q_jZYc9IIKUEHhClSTwZI4ClMeVvQS/view?usp=sharing) \| [shell](../tools/shells/erfnet_baseline_culane.sh) |
| Baseline | ENet | level 0 | 89.26 | 68.15 | 62.99 | 42.43 | 68.59 | 83.10 | 58.49 | 63.23 | 2464 | 69.90 | [model](https://drive.google.com/file/d/1DNgOpAVq87GIPUeAdMP6fnS4LhUmqyRB/view?usp=sharing) \| [shell](../tools/shells/enet_baseline_culane.sh) |
| Baseline | MobileNetV2 | level 0 | 87.82 | 65.09 | 61.46 | 38.15 | 57.34 | 79.29 | 55.89 | 60.29 | 2114 | 67.41 | [model](https://drive.google.com/file/d/1xTW24b0bW_tzeXQc0znHrMrkBK4BL7_t/view?usp=sharing) \| [shell](../tools/shells/mobilenetv2_baseline_culane.sh) |
| Baseline | MobileNetV3-Large | level 0 | 88.20 | 66.33 | 63.08 | 40.41 | 56.15 | 79.81 | 59.15 | 61.96 | 2304 | 68.42 | [model](https://drive.google.com/file/d/1JJ6gGcH6fAwR3UcGAnmdels5Vm8Bz48Q/view?usp=sharing) \| [shell](../tools/shells/mobilenetv3-large_baseline_culane.sh) |
| Baseline | RepVGG-A0 | level 0 | 89.74 | 67.68 | 65.21 | 42.51 | 67.85 | 83.13 | 60.86 | 63.63 | 2011 | 70.56 | [model](https://drive.google.com/file/d/1IJtM5LT0GTsHHlO0USLpuZLA_KyuLRd_/view?usp=sharing) \| [shell](../tools/shells/repvgg-a0_baseline_culane.sh) |
| Baseline | RepVGG-A1 | level 0 | 89.92 | 68.60 | 65.43 | 41.99 | 66.64 | 84.78 | 61.38 | 64.85 | 2127 | 70.85 | [model](https://drive.google.com/file/d/1cQMaXCww-a3mPssQK9iFzHJh6SinxcOo/view?usp=sharing) \| [shell](../tools/shells/repvgg-a1_baseline_culane.sh) |
| Baseline | RepVGG-B0 | level 0 | 90.86 | 69.32 | 66.68 | 43.53 | 67.83 | 85.43 | 59.80 | 66.47 | 2189 | 71.81 | [model](https://drive.google.com/file/d/1NR4n7N7mK3yKvRAWZUbtRYQ0xHM2vL60/view?usp=sharing) \| [shell](../tools/shells/repvgg-b0_baseline_culane.sh) |
| Baseline | RepVGG-B1g2 | level 0 | 90.85 | 69.31 | 67.94 | 43.81 | 68.45 | 85.85 | 60.64 | 67.69 | 2092 | 72.20 | [model](https://drive.google.com/file/d/1tKo69RroMYMn_v_C51BuHJQDSN0I7R-m/view?usp=sharing) \| [shell](../tools/shells/repvgg-b1g2_baseline_culane.sh) |
| Baseline | RepVGG-B2 | level 0 | 90.82 | 69.84 | 67.65 | 43.02 | 72.08 | 85.76 | 61.75 | 67.67 | 2000 | 72.33 | [model](https://drive.google.com/file/d/1_3sS5U20lTDIsq5jS4cev0kZaUER9NPH/view?usp=sharing) \| [shell](../tools/shells/repvgg-b2_baseline_culane.sh) |
| Baseline | Swin-Tiny | level 0 | 89.55 | 68.36 | 63.56 | 42.53 | 61.96 | 82.64 | 60.81 | 65.21 | 2813 | 69.90 | [model](https://drive.google.com/file/d/1YI_pHGVuuWYyTLeUgdZVDS14WoQjPon4/view?usp=sharing) \| [shell](../tools/shells/swin-tiny_baseline_culane.sh) |
| SCNN | VGG16 | level 0 | 92.02 | 72.31 | 69.13 | 46.01 | 76.37 | 87.71 | 64.68 | 68.96 | 1924 | 74.29 | [model](https://drive.google.com/file/d/1vm8B1SSH0nlAIbz3aEGC1kqWP4YdFb3A/view?usp=sharing) \| [shell](../tools/shells/vgg16_scnn_culane.sh) |
| SCNN | ResNet18 | level 0 | 90.98 | 70.17 | 66.54 | 43.12 | 66.31 | 85.62 | 62.20 | 65.58 | 1808 | 72.19 | [model](https://drive.google.com/file/d/1i08KOS3b0hOTuzn866j4oUWtw3TYcndn/view?usp=sharing) \| [shell](../tools/shells/resnet18_scnn_culane.sh) |
| SCNN | ResNet34 | level 0 | 91.06 | 70.41 | 67.75 | 44.64 | 68.98 | 86.50 | 61.57 | 65.75 | 2017 | 72.70 | [model](https://drive.google.com/file/d/1JyPJQv8gpFZbr1sh7zRRiekUuDR4Aea8/view?usp=sharing) \| [shell](../tools/shells/resnet34_scnn_culane.sh) |
| SCNN | ResNet50 | level 0 | 91.38 | 70.60 | 67.62 | 45.02 | 71.24 | 86.90 | 66.03 | 66.17 | 1958 | 73.03 | [model](https://drive.google.com/file/d/1DxqUONEpT47RvJlCg7kDWIsdkYH0Fv8E/view?usp=sharing) \| [shell](../tools/shells/resnet50_scnn_culane.sh) |
| SCNN | ResNet101 | level 0 | 91.10 | 71.43 | 68.53 | 46.39 | 72.61 | 86.87 | 61.95 | 67.01 | 1720 | 73.58 | [model](https://drive.google.com/file/d/11O4ZDvNqQsKodnl9kJar6Mx1UKnB70L9/view?usp=sharing) \| [shell](../tools/shells/resnet101_scnn_culane.sh) |
| SCNN | ERFNet | level 0 | 91.82 | 72.13 | 69.49 | 46.68 | 70.59 | 87.40 | 64.18 | 68.30 | 2236 | 74.03 | [model](https://drive.google.com/file/d/1YOAuIJqh0M1RsPN5zISY7kTx9xt29IS3/view?usp=sharing) \| [shell](../tools/shells/erfnet_scnn_culane.sh) |
| SCNN | RepVGG-A0 | level 0 | 91.06 | 71.30 | 67.23 | 44.75 | 70.51 | 87.11 | 61.73 | 66.61 | 1963 | 72.89 | [model](https://drive.google.com/file/d/1ayyVr5lVW5HxnF5QZWJl0IKW1nO5Q_KJ/view?usp=sharing) \| [shell](../tools/shells/repvgg-a1_scnn_culane.sh) |
| RESA | ResNet18 | level 0 | 91.23 | 70.57 | 67.16 | 45.24 | 68.01 | 86.56 | 64.32 | 66.19 | 1679 | 72.90 | [model](https://drive.google.com/file/d/1VkjZ1v-uMSy2VW8VdWPOOO1pFCXhfEDK/view?usp=sharing) \| [shell](../tools/shells/resnet18_resa_culane.sh) |
| RESA | ResNet34 | level 0 | 91.31 | 71.80 | 67.54 | 46.57 | 72.74 | 86.94 | 64.46 | 67.31 | 1701 | 73.66 | [model](https://drive.google.com/file/d/1x9JWhW7AIbiADqkzmKgmBUQL-JXexABM/view?usp=sharing) \| [shell](../tools/shells/resnet34_resa_culane.sh) |
| RESA | ResNet50 | level 0 | 91.52 | 72.49 | 68.44 | 47.02 | 72.56 | 87.34 | 63.11 | 68.21 | 1493 | 74.19 | [model](https://drive.google.com/file/d/1tmp5JO2CKWekbKVNX6nVKIj3AoQVCOkG/view?usp=sharing) \| [shell](../tools/shells/resnet50_resa_culane.sh) |
| RESA | ResNet101 | level 0 | 91.45 | 71.51 | 69.01 | 46.54 | 75.52 | 87.75 | 63.90 | 68.24 | 1522 | 74.04 | [model](https://drive.google.com/file/d/1RLLo8MUZDl4wahTXbn49FlxBq5IUsWru/view?usp=sharing) \| [shell](../tools/shells/resnet101_resa_culane.sh) |
| RESA | MobileNetV2 | level 0 | 90.58 | 70.42 | 67.19 | 45.29 | 62.80 | 85.52 | 66.00 | 65.19 | 1945 | 72.36 | [model](https://drive.google.com/file/d/1Dh9Laid33aMGFBQuTqasKgqluWKTU5Si/view?usp=sharing) \| [shell](../tools/shells/mobilenetv2_resa_culane.sh) |
| RESA | MobileNetV3-Large | level 0 | 89.53 | 67.63 | 65.74 | 43.08 | 66.07 | 84.61 | 60.10 | 63.14 | 2218 | 70.61 | [model](https://drive.google.com/file/d/1mgqkDgss9nDQgAOQnNIoHuprxofod1V4/view?usp=sharing) \| [shell](../tools/shells/mobilenetv3-large_resa_culane.sh) |
| LSTR | ResNet18s-2X | level 0 | 56.17 | 39.10 | 22.90 | 25.62 | 25.49 | 52.09 | 40.21 | 30.33 | 1690 | 39.77 | [model](https://drive.google.com/file/d/1vdYwM0xDcQLjMAibjmls8hX-IsUe0xcq/view?usp=sharing) \| [shell](../tools/shells/resnet18s_lstr_culane.sh) |
| LSTR | ResNet18s-2X | level 1a | 86.78 | 67.34 | 59.92 | 40.10 | 59.82 | 78.66 | 56.63 | 56.64 | 1166 | 68.72 | [model](https://drive.google.com/file/d/11Tv_nowlWmQtTYQfhGsziDIzb20kPo8o/view?usp=sharing) \| [shell](../tools/shells/resnet18s_lstr-aug_culane.sh) |
| LSTR | ResNet34 | level 1a | 89.73 | 69.77 | 66.72 | 45.32 | 68.16 | 85.03 | 64.34 | 64.13 | 1247 | 72.48 | [model](https://drive.google.com/file/d/1KfmXubuAtUoE9MO8iViMyB_3XhTxWnwH/view?usp=sharing) \| [shell](../tools/shells/resnet34_lstr-aug_culane.sh) |
| BezierLaneNet | ResNet18 | level 1b | 90.22 | 71.55 | 68.70 | 45.30 | 70.91 | 84.09 | 62.49 | 58.98 | 996 | 73.67 | [model](https://drive.google.com/file/d/1IpfusHvFeMEGe8wv0fer6KF3pH4X2Tj3/view?usp=sharing) \| [shell](../tools/shells/resnet18_bezierlanenet_culane-aug1b.sh) |
| BezierLaneNet | ResNet34 | level 1b | 91.59 | 73.20 | 69.90 | 48.05 | 76.74 | 87.16 | 69.20 | 62.45 | 888 | 75.57 | [model](https://drive.google.com/file/d/1342FQeDQKRHMo283jW2T1WDgfgsYbR5q/view?usp=sharing) \| [shell](../tools/shells/resnet34_bezierlanenet_culane-aug1b.sh) |

### LLAMAS detailed performance (best):

| method | backbone | data<br>augmentation | F1 | TP | FP | FN | Precision | Recall | val / test | | 
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | 
| Baseline | VGG16 | level 0 | 95.11 | 70263 | 3460 | 3772 | 95.31 | 94.91 | val | [model](https://drive.google.com/file/d/1k2b7iRw3_YMJDMUsKVjtdOXRdZsXNXnO/view?usp=sharing) \| [shell](../tools/shells/vgg16_baseline_llamas.sh) |
| Baseline | ResNet34 | level 0 | 95.91 | 70841 | 2847 | 3194 | 96.14 | 95.69 | val | [model](https://drive.google.com/file/d/1YXNgwhQqwxoMkHDbRqAdnuAXIe1IdLSm/view?usp=sharing) \| [shell](../tools/shells/resnet34_baseline_llamas.sh) |
| Baseline | ERFNet | level 0 | 96.13 | 71136 | 2830 | 2899 | 96.17 | 96.08 | val | [model](https://drive.google.com/file/d/15oNU4iffIPYTuKSCH2j0boRkYmm3-uSh/view?usp=sharing) \| [shell](../tools/shells/erfnet_baseline_llamas.sh) |
| SCNN | VGG16 | level 0 | 96.42 | 71274 | 2526 | 2761 | 96.27 | 96.42 | val | [model](https://drive.google.com/file/d/1qE-euGGMZTxcHED_VDUW6eR-SKPkUDD-/view?usp=sharing) \| [shell](../tools/shells/vgg16_scnn_llamas.sh) |
| SCNN | ERFNet | level 0 | 95.94 | 71036 | 3019 | 2999 | 95.92 | 95.95 | val | [model](https://drive.google.com/file/d/1oTdmP_tsguqa1-6bBIikT4gPThUVdXCr/view?usp=sharing) \| [shell](../tools/shells/erfnet_scnn_llamas.sh) |
| SCNN | ResNet34 | level 0 | 96.19 | 71109 | 2705 | 2926 | 96.34 | 96.05 | val | [model](https://drive.google.com/file/d/1-vribu32iXViBqYumApmKQK4mjQd6BEp/view?usp=sharing) \| [shell](../tools/shells/resnet34_scnn_llamas.sh) |
| BezierLaneNet | ResNet18 | level 1b | 95.52 | 70515 | 3102 | 3520 | 95.79 | 95.25 | val | [model](https://drive.google.com/file/d/1fTQEZnr2wVQ20P3B2AyM3c_dFp5BHKwQ/view?usp=sharing) \| [shell](../tools/shells/resnet18_bezierlanenet_llamas-aug1b.sh) |
| BezierLaneNet | ResNet34 | level 1b | 96.11 | 70959 | 2667 | 3076 | 96.38 | 95.85 | val | [model](https://drive.google.com/file/d/1RhYTJB_VlHL9hFYuwAX_T4Nev9ZIlmHt/view?usp=sharing) \| [shell](../tools/shells/resnet34_bezierlanenet_llamas-aug1b.sh) |

Their test performance can be found at the [LLAMAS leaderboard](https://unsupervised-llamas.com/llamas/benchmark_splines).

## Semantic segmentation performance

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
