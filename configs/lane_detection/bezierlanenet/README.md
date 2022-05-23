# BézierLaneNet

> [Rethinking Efficient Lane Detection via Curve Modeling](https://arxiv.org/abs/2203.02431) **CVPR 2022**

## Method Overview

State-of-the-art lane detectors are typically based on semantic segmentation (SCNN, RESA) or point detection (LaneATT). However, semantic segmentation requires customized post-processing & cannot deal with a variable number of lanes. Point detection methods are currently anchor-based, with NMS as post-processing. The more natural way would be getting a curve representation directly. Methods like LSTR made the first steps in this direction, but didn't really achieved comparable performance against SOTA methods. BézierLaneNet use a fully convolutional network to predict cubic Bézier curves, the ease of optimization of Bézier control points made it possible for direct curve methods to compete with SOTAs. A fusion of flipped feature maps is also employed to exploit symmetry in the car's front-view. BézierLaneNet (ResNet-34) achieves 75.6 F-1 on CULane, and attained the 1st place (of all published methods) in the LLAMAS leaderboard at its time, while running at 150 FPS in our benchmark.

<div align=center>
<img src="https://user-images.githubusercontent.com/32259501/157155447-81f28ec6-3ebe-42e0-8864-c739d8c44155.png"/>
</div>

For another earlier attempt on learning Bézier curves for lane detection with (almost) the same name BezierLaneNet, please refer to [wiki](https://github.com/voldemortX/pytorch-auto-drive/wiki/Notes) **9. BézierLaneNet disclaimer** and [this repo](https://github.com/mo-vic/BezierLaneNet).

## Results

*Training time estimated with single 2080 Ti.*

*ImageNet pre-training, 3-times average/best.*

### TuSimple (test)

| backbone | aug | resolution | training time | precision | accuracy (avg) | accuracy |  FP | FN | |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| ResNet18 | level 1b | 360 x 640 | 5.5h | full | 95.01% | 95.41% | 0.0531 | 0.0458 | [model](https://drive.google.com/file/d/10qMdvPBnZP4P88EQXYZxsXZgj7sz6LvS/view?usp=sharing) \| [shell](/tools/shells/resnet18_bezierlanenet_tusimple-aug1b.sh) |
| ResNet34 | level 1b | 360 x 640 | 6.5h | full | 95.17% | 95.65% | 0.0513 | 0.0386 | [model](https://drive.google.com/file/d/1FFn8j2BoUsyj8UbBcfeGWKvCQj9Qg-44/view?usp=sharing) \| [shell](/tools/shells/resnet34_bezierlanenet_tusimple-aug1b.sh) |

### CULane (test)

| backbone | aug | resolution | training time | precision | F1 (avg) | F1 | normal | crowded | night | no line | shadow | arrow | dazzle<br>light | curve | crossroad | |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| ResNet18 | level 1b | 288 x 800 | 9.9h | mix | 73.36 | 73.67 | 90.22 | 71.55 | 68.70 | 45.30 | 70.91 | 84.09 | 62.49 | 58.98 | 996 | [model](https://drive.google.com/file/d/1IpfusHvFeMEGe8wv0fer6KF3pH4X2Tj3/view?usp=sharing) \| [shell](/tools/shells/resnet18_bezierlanenet_culane-aug1b.sh) |
| ResNet34 | level 1b | 288 x 800 | 11.0h | mix | 75.30 | 75.57 | 91.59 | 73.20 | 69.90 | 48.05 | 76.74 | 87.16 | 69.20 | 62.45 | 888 | [model](https://drive.google.com/file/d/1342FQeDQKRHMo283jW2T1WDgfgsYbR5q/view?usp=sharing) \| [shell](/tools/shells/resnet34_bezierlanenet_culane-aug1b.sh) |

### LLAMAS (val)

| backbone | aug | resolution | training time | precision | F1 (avg) | F1 | TP | FP | FN | Precision | Recall | |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| ResNet18 | level 1b | 360 x 640 | 5.5h | mix | 95.42 | 95.52 | 70515 | 3102 | 3520 | 95.79 | 95.25 | [model](https://drive.google.com/file/d/1fTQEZnr2wVQ20P3B2AyM3c_dFp5BHKwQ/view?usp=sharing) \| [shell](/tools/shells/resnet18_bezierlanenet_llamas-aug1b.sh) |
| ResNet34 | level 1b | 360 x 640 | 6.1h | mix | 96.04 | 96.11 | 70959 | 2667 | 3076 | 96.38 | 95.85 | [model](https://drive.google.com/file/d/1RhYTJB_VlHL9hFYuwAX_T4Nev9ZIlmHt/view?usp=sharing) \| [shell](/tools/shells/resnet34_bezierlanenet_llamas-aug1b.sh) |

Their test performance can be found at the [LLAMAS leaderboard](https://unsupervised-llamas.com/llamas/benchmark_splines).

## Profiling

*FPS is best trial-avg among 3 trials on a 2080 Ti.*

| backbone | resolution | FPS | FLOPS(G) | Params(M) |
| :---: | :---: | :---: | :---: | :---: |
| ResNet18 | 360 x 640 | 212.83 | 14.77 | 4.10 |
| ResNet34 | 360 x 640 | 149.52 | 29.85 | 9.49 |
| ResNet18 | 288 x 800 | 210.79 | 14.66 | 4.10 |
| ResNet34 | 288 x 800 | 144.65 | 29.54 | 9.49 |

## Citation

```
@inproceedings{feng2022rethinking,
  title={Rethinking efficient lane detection via curve modeling},
  author={Feng, Zhengyang and Guo, Shaohua and Tan, Xin and Xu, Ke and Wang, Min and Ma, Lizhuang},
  booktitle={Computer Vision and Pattern Recognition},
  year={2022}
}
```
