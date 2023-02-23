# PytorchAutoDrive: Framework for self-driving perception


*PytorchAutoDrive* is a **pure Python** framework includes semantic segmentation models, lane detection models based on **PyTorch**. Here we provide full stack supports from research (model training, testing, fair benchmarking by simply writing configs) to application (visualization, model deployment).

**Paper:** [Rethinking Efficient Lane Detection via Curve Modeling](https://arxiv.org/abs/2203.02431) (CVPR 2022)

**Poster:** [PytorchAutoDrive: Toolkit & Fair Benchmark for Autonomous Driving Research](https://drive.google.com/file/d/14EgcwPnKvAZJ1aWqBv6W9Msm666Wqi5a/view?usp=sharing) (PyTorch Developer Day 2021)

*This repository is under active development, results with models uploaded are stable. For legacy code users, please check [deprecations](https://github.com/voldemortX/pytorch-auto-drive/issues/14) for changes.*

**A demo video from ERFNet:**

https://user-images.githubusercontent.com/32259501/148680744-a18793cd-f437-461f-8c3a-b909c9931709.mp4

## Highlights

Various methods on a wide range of backbones, **config** based implementations, **modulated** and **easily understood** codes, image/keypoint loading, transformations and **visualizations**, **mixed precision training**, tensorboard logging and **deployment support** with ONNX and TensorRT.

Models from this repo are faster to train (**single card trainable**) and often have better performance than other implementations, see [wiki](https://github.com/voldemortX/pytorch-auto-drive/wiki/Notes) for reasons and technical specification of models.

## Supported datasets: 

| Task | Dataset |
| :---: | :---: |
| semantic segmentation | PASCAL VOC 2012 |
| semantic segmentation | Cityscapes |
| semantic segmentation | GTAV* |
| semantic segmentation | SYNTHIA* |
| lane detection | CULane |
| lane detection | TuSimple |
| lane detection | LLAMAS |
| lane detection | BDD100K (*In progress*) |

\* The UDA baseline setup, with Cityscapes *val* set as validation.

## Supported models:

| Task | Backbone | Model/Method |
| :---: | :---: | :---: |
| semantic segmentation | ResNet-101 | [FCN](/configs/semantic_segmentation/fcn) |
| semantic segmentation | ResNet-101 | [DeeplabV2](https://arxiv.org/abs/1606.00915) |
| semantic segmentation | ResNet-101 | [DeeplabV3](https://arxiv.org/abs/1706.05587) |
| semantic segmentation | - | [ENet](https://arxiv.org/abs/1606.02147) |
| semantic segmentation | - | [ERFNet](/configs/semantic_segmentation/erfnet) |
| lane detection | ENet, ERFNet, VGG16, ResNets (18, 34, 50, 101), MobileNets (V2, V3-Large), RepVGGs (A0, A1, B0, B1g2, B2), Swin (Tiny) | [Baseline](/configs/lane_detection/baseline) |
| lane detection | ERFNet, VGG16, ResNets (18, 34, 50, 101), RepVGGs (A1) | [SCNN](https://arxiv.org/abs/1712.06080) |
| lane detection | ResNets (18, 34, 50, 101), MobileNets (V2, V3-Large), ERFNet | [RESA](https://arxiv.org/abs/2008.13719) |
| lane detection | ERFNet, ENet | [SAD](https://arxiv.org/abs/1908.00821) ([*Postponed*](https://github.com/voldemortX/pytorch-auto-drive/wiki/Notes)) |
| lane detection | ERFNet | [PRNet](http://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123630698.pdf) (*In progress*) |
| lane detection | ResNets (18, 34, 50, 101), ResNet18-reduced | [LSTR](https://arxiv.org/abs/2011.04233) |
| lane detection | ResNets (18, 34) | [LaneATT](https://arxiv.org/abs/2010.12035) |
| lane detection | ResNets (18, 34) | [BézierLaneNet](/configs/lane_detection/bezierlanenet) |

## Model Zoo

We provide solid results (average/best/detailed), training time, shell scripts and trained models available for download in [MODEL_ZOO.md](docs/MODEL_ZOO_J.md).

## Installation

Please prepare the environment and code with [INSTALL.md](docs/INSTALL.md). Then follow the instructions in [DATASET.md](docs/DATASET.md) to set up datasets. 

## Getting Started

Get started with [LANEDETECTION.md](docs/LANEDETECTION.md) for lane detection.

Get started with [SEGMENTATION.md](docs/SEGMENTATION.md) for semantic segmentation.

## Visualization Tools

Refer to [VISUALIZATION.md](docs/VISUALIZATION.md) for a visualization & inference tutorial, for image and video inputs.

## Benchmark Tools

Refer to [BENCHMARK.md](docs/BENCHMARK.md) for a benchmarking tutorial, including FPS test, FLOPs & memory count for each supported model.

## Deployment

Refer to [DEPLOY.md](docs/DEPLOY.md) for ONNX and TensorRT deployment supports.

## Advanced Tutorial

Checkout [ADVANCED_TUTORIAL.md](docs/ADVANCED_TUTORIAL.md) for advanced use cases and how to code in PytorchAutoDrive.

## Contributing

Refer to [CONTRIBUTING.md](/docs/CONTRIBUTING.md) for contribution guides.

## Citation

If you feel this framework substantially helped your research or you want a reference when using our results, please cite the following paper that made the official release of PytorchAutoDrive:

```
@inproceedings{feng2022rethinking,
  title={Rethinking efficient lane detection via curve modeling},
  author={Feng, Zhengyang and Guo, Shaohua and Tan, Xin and Xu, Ke and Wang, Min and Ma, Lizhuang},
  booktitle={Computer Vision and Pattern Recognition},
  year={2022}
}
```

## Credits:

PytorchAutoDrive is maintained by Zhengyang Feng ([voldemortX](https://github.com/voldemortX)) and Shaohua Guo ([cedricgsh](https://github.com/cedricgsh)).

Contributors (GitHub ID): [kalkun](https://github.com/kalkun), [LittleJohnKhan](https://github.com/LittleJohnKhan), [francis0407](https://github.com/francis0407), [PannenetsF](https://github.com/PannenetsF)

People who sponsored us (e.g., with hardware): [Lizhuang Ma](https://dmcv.sjtu.edu.cn/people/), [Xin Tan](https://tanxincs.github.io/TAN-Xin.github.io/), Junshu Tang ([junshutang](https://github.com/junshutang)), Fengqi Liu ([FengqiLiu1221](https://github.com/FengqiLiu1221)) 
