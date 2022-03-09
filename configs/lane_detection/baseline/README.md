# Baseline

> Not exactly proposed by any paper.

## Method Overview

The segmentation baseline takes semantic segmentation networks and appends a lane existence head. It takes the most classic multi-class segmentation approach, its design originates from the SCNN paper (ResNets and VGG based Deeplab), while the SAD paper explores the use of ENet and ERFNet, later the RESA paper reduced the network width for efficient ResNet baselines, finally the BézierLaneNet paper (this framework) improved these baselines with modern training techniques and fair evaluations, further extended them to modern architectures such as Swin Transformer, RepVGG and MobileNets. Among them, the ERFNet baseline even achieves comparable performance against SOTA methods. However, they are very sensitive to hyper-parameters, see [Wiki](https://github.com/voldemortX/pytorch-auto-drive/wiki/Notes) and the BézierLaneNet Appendix.B for more info. Specifically, the VGG16 backbone corresponds to DeepLab-LargeFOV in SCNN, the ResNet & other backbones correspond to DeepLabV2 (w.o. ASPP) with output channels reduced to 128 as in RESA. We sometimes call them by backbone names for consistency with common practices.

## Results

*Training time estimated with single 2080 Ti.*

*ImageNet pre-training, 3-times average/best.*

*<sup>+</sup> Measured on a single GTX 1080Ti.*

*<sup>#</sup> No pre-training.*

*\* Trained on a 1080 Ti cluster, with CUDA 9.0 PyTorch 1.3, training time is estimated as: single 2080 Ti, mixed precision.*

*\*\* Trained on two 2080ti.*  

### TuSimple (test)

| backbone | aug | resolution | training time | precision | accuracy (avg) | accuracy |  FP | FN | |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| VGG16 | level 0 | 360 x 640 | 1.5h | mix | 93.79% | 93.94% | 0.0998 | 0.1021 | [model](https://drive.google.com/file/d/1ChK0hApqLU0xUiEm4Wul-gNYQDQka151/view?usp=sharing) \| [shell](/tools/shells/vgg16_baseline_tusimple.sh) |
| ResNet18 | level 0 | 360 x 640 | 0.7h | mix | 94.18% | 94.25% | 0.0881 | 0.0894 | [model](https://drive.google.com/file/d/17VKnwsN4WMbpnD4DgaaerppjXybqn-LG/view?usp=sharing) \| [shell](/tools/shells/resnet18_baseline_tusimple.sh) |
| ResNet34 | level 0 | 360 x 640 | 1.1h | mix | 95.23% | 95.31% | 0.0640 | 0.0622 | [model](https://drive.google.com/file/d/1NAck0aQZK_wAHer4xB8xzegxDWk9EFtG/view?usp=sharing) \| [shell](/tools/shells/resnet34_baseline_tusimple.sh) |
| ResNet34 | level 1a | 360 x 640 | 1.2h* | full | 92.14% | 92.68% | 0.1073 | 0.1221 | [model](https://drive.google.com/file/d/1OhN2tWIep9ncKFf-_2RqUEaSJvPK60cn/view?usp=sharing) \| [shell](/tools/shells/resnet34_baseline-aug_tusimple.sh) |
| ResNet50 | level 0 | 360 x 640 | 1.5h | mix | 95.07% | 95.12% | 0.0649 | 0.0653 | [model](https://drive.google.com/file/d/10KBMVGc63kPvqL_2deaLfTfC3fSAtnju/view?usp=sharing) \| [shell](/tools/shells/resnet50_baseline_tusimple.sh) |
| ResNet101 | level 0 | 360 x 640 | 2.6h | mix | 95.15% | 95.19% | 0.0619 | 0.0620 | [model](https://drive.google.com/file/d/1mELtKB3e8ntOmPovhnMphXWKf_bv83ef/view?usp=sharing) \| [shell](/tools/shells/resnet101_baseline_tusimple.sh) |
| ERFNet | level 0 | 360 x 640 | 0.8h | mix | 96.02% | 96.04% | 0.0591 | 0.0365 | [model](https://drive.google.com/file/d/1rLWDP_dkIQ7sBsCEzJi8T7ET1EPghhJJ/view?usp=sharing) \| [shell](/tools/shells/erfnet_baseline_tusimple.sh) |
| ERFNet | level 1a | 360 x 640 | 0.9h* | full | 94.21% | 94.37% | 0.0846 | 0.0770 | [model](https://drive.google.com/file/d/1LPmxT8rnyZL2M08lSLrlvrM0H_hMrFvq/view?usp=sharing) \| [shell](/tools/shells/erfnet_baseline-aug_tusimple.sh) |
| ENet<sup>#</sup> | level 0 | 360 x 640 | 1h<sup>+</sup> | mix | 95.55% | 95.61% | 0.0655 | 0.0503 | [model](https://drive.google.com/file/d/1CNSox62ghs0ArDVJb9mTZ1NVvqSkUNYC/view?usp=sharing) \| [shell](/tools/shells/enet_baseline_tusimple.sh) |
| MobileNetV2 | level 0 | 360 x 640 | 0.5h | mix | 93.98% | 94.07% | 0.0792 | 0.0866 | [model](https://drive.google.com/file/d/1SUqt3BDXSMhAv68F9VIKncY0lDUg9My8/view?usp=sharing) \| [shell](/tools/shells/mobilenetv2_baseline_tusimple.sh) |
| MobileNetV3-Large | level 0 | 360 x 640 | 0.5h | mix | 92.09% | 92.18% | 0.1149 | 0.1322 | [model](https://drive.google.com/file/d/1I5SPlkmC8TnNeANoQGxzP3P1_iVxms3u/view?usp=sharing) \| [shell](/tools/shells/mobilenetv3-large_baseline_tusimple.sh) |

### CULane (test)

| backbone | aug | resolution | training time | precision | F measure (avg) | F measure | normal | crowded | night | no line | shadow | arrow | dazzle<br>light | curve | crossroad | |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| VGG16 | level 0 | 288 x 800 | 9.3h | mix | 65.93 | 66.09 | 85.51 | 64.05 | 61.14 | 35.96 | 59.76 | 78.43 | 53.25 | 62.16 | 2224 | [model](https://drive.google.com/file/d/1wVz1a7S1e5Dgy7ERk7E8dqQ8gyK-dWLG/view?usp=sharing) \| [shell](/tools/shells/vgg16_baseline_culane.sh) |
| ResNet18 | level 0 | 288 x 800 | 5.3h | mix | 65.19 | 65.30 | 85.45 | 62.63 | 61.04 | 33.88 | 51.72 | 78.15 | 53.05 | 59.70 | 1915 | [model](https://drive.google.com/file/d/1wkaTp8v1ceXrd6AjRccqpNxxxkd_qg1U/view?usp=sharing) \| [shell](/tools/shells/resnet18_baseline_culane.sh) |
| ResNet34 | level 0 | 288 x 800 | 7.3h | mix | 69.82 | 69.92 | 89.46 | 66.66 | 65.38 | 40.43 | 62.17 | 83.18 | 58.51 | 63.00 | 1713 | [model](https://drive.google.com/file/d/16VIJcd3wDOjFjg3UCVekUPcAb_F1K604/view?usp=sharing) \| [shell](/tools/shells/resnet34_baseline_culane.sh) |
| ResNet50 | level 0 | 288 x 800 | 12.4h | mix | 68.31 | 68.48 | 88.15 | 65.73 | 63.74 | 37.96 | 62.59 | 81.68 | 59.47 | 64.01 | 2046 | [model](https://drive.google.com/file/d/1DYVeH9kdSPhEMA4fsJFdEiw8qOwvQBl8/view?usp=sharing) \| [shell](/tools/shells/resnet50_baseline_culane.sh) |
| ResNet101 | level 0 | 288 x 800 | 20.0h | mix | 71.29 | 71.37 | 90.11 | 67.89 | 67.01 | 43.10 | 70.56 | 85.09 | 61.77 | 65.47 | 1883 | [model](https://drive.google.com/file/d/1iubFjWetsKE2VI4BEIWLDd80gB7IQUaP/view?usp=sharing) \| [shell](/tools/shells/resnet101_baseline_culane.sh) |
| ERFNet | level 0 | 288 x 800 | 6h | mix | 73.40 | 73.49 | 91.48 | 71.27 | 68.09 | 46.76 | 74.47 | 86.09 | 64.18 | 66.89 | 2102 | [model](https://drive.google.com/file/d/16-Q_jZYc9IIKUEHhClSTwZI4ClMeVvQS/view?usp=sharing) \| [shell](/tools/shells/erfnet_baseline_culane.sh) |
| ENet<sup>#</sup> | level 0 | 288 x 800 |  6.4h<sup>+</sup> | mix | 69.39 | 69.90 | 89.26 | 68.15 | 62.99 | 42.43 | 68.59 | 83.10 | 58.49 | 63.23 | 2464 | [model](https://drive.google.com/file/d/1DNgOpAVq87GIPUeAdMP6fnS4LhUmqyRB/view?usp=sharing) \| [shell](/tools/shells/enet_baseline_culane.sh) |
| MobileNetV2 | level 0 | 288 x 800 | 3.0h | mix | 67.34 | 67.41 | 87.82 | 65.09 | 61.46 | 38.15 | 57.34 | 79.29 | 55.89 | 60.29 | 2114 | [model](https://drive.google.com/file/d/1xTW24b0bW_tzeXQc0znHrMrkBK4BL7_t/view?usp=sharing) \| [shell](/tools/shells/mobilenetv2_baseline_culane.sh) |
| MobileNetV3-Large | level 0 | 288 x 800 | 3.0h | mix | 68.27 | 68.42 | 88.20 | 66.33 | 63.08 | 40.41 | 56.15 | 79.81 | 59.15 | 61.96 | 2304 | [model](https://drive.google.com/file/d/1JJ6gGcH6fAwR3UcGAnmdels5Vm8Bz48Q/view?usp=sharing) \| [shell](/tools/shells/mobilenetv3-large_baseline_culane.sh) |
| RepVGG-A0 | level 0 | 288 x 800 | 3.3h** | mix | 70.22 | 70.56 | 89.74 | 67.68 | 65.21 | 42.51 | 67.85 | 83.13 | 60.86 | 63.63 | 2011 | [model](https://drive.google.com/file/d/1IJtM5LT0GTsHHlO0USLpuZLA_KyuLRd_/view?usp=sharing) \| [shell](/tools/shells/repvgg-a0_baseline_culane.sh) |
| RepVGG-A1 | level 0 | 288 x 800 | 4.1h** | mix | 70.73 | 70.85 | 89.92 | 68.60 | 65.43 | 41.99 | 66.64 | 84.78 | 61.38 | 64.85 | 2127 | [model](https://drive.google.com/file/d/1cQMaXCww-a3mPssQK9iFzHJh6SinxcOo/view?usp=sharing) \| [shell](/tools/shells/repvgg-a1_baseline_culane.sh) |
| RepVGG-B0 | level 0 | 288 x 800 | 6.2h** | mix | 71.77 | 71.81 | 90.86 | 69.32 | 66.68 | 43.53 | 67.83 | 85.43 | 59.80 | 66.47 | 2189 | [model](https://drive.google.com/file/d/1NR4n7N7mK3yKvRAWZUbtRYQ0xHM2vL60/view?usp=sharing) \|  [shell](/tools/shells/repvgg-b0_baseline_culane.sh) |
| RepVGG-B1g2 | level 0 | 288 x 800 | 10.0h** | mix | 72.08 | 72.20 | 90.85 | 69.31 | 67.94 | 43.81 | 68.45 | 85.85 | 60.64 | 67.69 | 2092 | [model](https://drive.google.com/file/d/1tKo69RroMYMn_v_C51BuHJQDSN0I7R-m/view?usp=sharing) \| [shell](/tools/shells/repvgg-b1g2_baseline_culane.sh) |
| RepVGG-B2 | level 0 | 288 x 800 | 13.2h** | mix | 72.24 | 72.33 | 90.82 | 69.84 | 67.65 | 43.02 | 72.08 | 85.76 | 61.75 | 67.67 | 2000 | [model](https://drive.google.com/file/d/1_3sS5U20lTDIsq5jS4cev0kZaUER9NPH/view?usp=sharing) \| [shell](/tools/shells/repvgg-b2_baseline_culane.sh) |
| Swin-Tiny | level 0 | 288 x 800 | 12.1h** | mix | 69.75 | 69.90 | 89.55 | 68.36 | 63.56 | 42.53 | 61.96 | 82.64 | 60.81 | 65.21 | 2813 | [model](https://drive.google.com/file/d/1YI_pHGVuuWYyTLeUgdZVDS14WoQjPon4/view?usp=sharing) \| [shell](/tools/shells/swin-tiny_baseline_culane.sh) |

### LLAMAS (val)

| backbone | aug | resolution | training time | precision | F1 | F1 | TP | FP | FN | Precision | Recall | |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| VGG16 | level 0 | 360 x 640 | 9.3h | mix | 95.05 | 95.11 | 70263 | 3460 | 3772 | 95.31 | 94.91 | [model](https://drive.google.com/file/d/1k2b7iRw3_YMJDMUsKVjtdOXRdZsXNXnO/view?usp=sharing) \| [shell](/tools/shells/vgg16_baseline_llamas.sh) |
| ResNet34 | level 0 | 360 x 640 | 7.0h | mix | 95.90 | 95.91 | 70841 | 2847 | 3194 | 96.14 | 95.69 | [model](https://drive.google.com/file/d/1YXNgwhQqwxoMkHDbRqAdnuAXIe1IdLSm/view?usp=sharing) \| [shell](/tools/shells/resnet34_baseline_llamas.sh) |
| ERFNet | level 0 | 360 x 640 | 10.9h<sup>+</sup> | mix | 95.94 | 96.13 | 71136 | 2830 | 2899 | 96.17 | 96.08 | [model](https://drive.google.com/file/d/15oNU4iffIPYTuKSCH2j0boRkYmm3-uSh/view?usp=sharing) \| [shell](/tools/shells/erfnet_baseline_llamas.sh) |

Their test performance can be found at the [LLAMAS leaderboard](https://unsupervised-llamas.com/llamas/benchmark_splines).

## Profiling

*FPS is best trial-avg among 3 trials on a 2080 Ti. Post-processing is ignored.*

| backbone | resolution | FPS | FLOPS(G) | Params(M) |
| :---: | :---: | :---: | :---: | :---: |
| VGG16 | 360 x 640 | 56.36 | 214.50 | 20.37 | 
| ResNet18 | 360 x 640 | 148.59 | 85.24 | 12.04 | 
| ResNet34 | 360 x 640 | 79.97 | 159.60 | 22.15 |
| ResNet50 | 360 x 640 | 50.58 | 177.62 | 24.57 |
| ResNet101 | 360 x 640 | 27.41 | 314.36 | 43.56 |
| ERFNet | 360 x 640 | 85.87 | 26.32 | 2.67 | 
| ENet | 360 x 640 | 56.63 | 4.26 | 0.95 |
| MobileNetV2 | 360 x 640 | 126.54 | 4.49 | 2.06 |
| MobileNetV3-Large | 360 x 640 | 104.34 | 3.63 | 3.30 |
| VGG16 | 288 x 800 | 55.31 | 214.50 | 20.15 | 
| ResNet18 | 288 x 800 | 136.28 | 85.22 | 11.82 | 
| ResNet34 | 288 x 800 | 72.42 | 159.60 | 21.93 | 
| ResNet50 | 288 x 800 | 49.41 | 177.60 | 24.35 | 
| ResNet101 | 288 x 800 | 27.19 | 314.34 | 43.34 | 
| ERFNet | 288 x 800 | 88.76 | 26.26 | 2.68 | 
| ENet | 288 x 800 | 57.99 | 4.12 | 0.96 |
| MobileNetV2 | 288 x 800 | 129.24 | 4.41 | 2.00 |
| MobileNetV3-Large | 288 x 800 | 107.83 | 3.56 | 3.25 |
| RepVGG-A0 | 288 x 800 | 162.61 | 207.81 | 9.06 |
| RepVGG-A1 | 288 x 800 | 117.30 | 339.83 | 13.54 |
| RepVGG-B0 | 288 x 800 | 103.68 | 390.83 | 15.09 |
| RepVGG-B1g2 | 288 x 800 | 36.91 | 1166.76 | 42.20 |
| RepVGG-B2 | 288 x 800 | 18.98 | 2310.13 | 81.23 |
| Swin-Tiny | 288 x 800 | 51.90 | 44.24 | 27.72 |

## Citation (if you have to)

```
@inproceedings{pan2018spatial,
  title={Spatial as deep: Spatial cnn for traffic scene understanding},
  author={Pan, Xingang and Shi, Jianping and Luo, Ping and Wang, Xiaogang and Tang, Xiaoou},
  booktitle={AAAI},
  year={2018}
}

@inproceedings{feng2022rethinking,
  title={Rethinking efficient lane detection via curve modeling},
  author={Feng, Zhengyang and Guo, Shaohua and Tan, Xin and Xu, Ke and Wang, Min and Ma, Lizhuang},
  booktitle={Computer Vision and Pattern Recognition},
  year={2022}
}
```
