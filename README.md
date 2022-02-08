# Codebase for deep self-driving perception

*PytorchAutoDrive* is a **Python** codebase includes semantic segmentation models, lane detection models based on **PyTorch**. Here we provide full stack supports from research (model training, testing, fair benchmarking by simply writing configs) to application (visualization, model deployment). Poster at PyTorch Developer Day: [PytorchAutoDrive: Toolkit & Fair Benchmark for Autonomous Driving Research](https://drive.google.com/file/d/14EgcwPnKvAZJ1aWqBv6W9Msm666Wqi5a/view?usp=sharing).

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
| semantic segmentation | ResNet-101 | [FCN](https://arxiv.org/abs/1605.06211) |
| semantic segmentation | ResNet-101 | [DeeplabV2](https://arxiv.org/abs/1606.00915) |
| semantic segmentation | ResNet-101 | [DeeplabV3](https://arxiv.org/abs/1706.05587) |
| semantic segmentation | - | [ENet](https://arxiv.org/abs/1606.02147) |
| semantic segmentation | - | [ERFNet](https://ieeexplore.ieee.org/abstract/document/8063438/) |
| lane detection | ENet, ERFNet, VGG16, ResNets (18, 34, 50, 101), RepVGGs (A0, A1, B0, B1g2, B2), Swin (Tiny) | Baseline |
| lane detection | ERFNet, VGG16, ResNets (18, 34, 50, 101), RepVGGs (A1) | [SCNN](https://arxiv.org/abs/1712.06080) |
| lane detection | ResNets (18, 34, 50, 101) | [RESA](https://arxiv.org/abs/2008.13719) |
| lane detection | ERFNet, ENet | [SAD](https://arxiv.org/abs/1908.00821) ([*Postponed*](https://github.com/voldemortX/pytorch-auto-drive/wiki/Notes)) |
| lane detection | ERFNet | [PRNet](http://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123630698.pdf) (*In progress*) |
| lane detection | ResNets (18, 34, 50, 101), ResNet18-reduced | [LSTR](https://arxiv.org/abs/2011.04233) |

*The VGG16 backbone corresponds to DeepLab-LargeFOV in SCNN.*

*The ResNet backbone corresponds to DeepLabV2 (w.o. ASPP) with output channels reduced to 128 as in RESA.*

*We keep calling it VGG16/ResNet for consistency with common practices.*

## Model Zoo

We provide solid results (average/best/detailed), training time, shell scripts and trained models available for download in [MODEL_ZOO.md](docs/MODEL_ZOO.md).

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

We welcome **Pull Requests** to fix bugs, update docs or implement new features etc. We also welcome **Issues** to report problems and needs, or ask questions (since your question might be more common and helpful to the community than you presume). Interested folks should checkout our [roadmap](https://github.com/voldemortX/pytorch-auto-drive/issues/4).

This repository implements (or plan to implement) the following interesting papers in a unified PyTorch codebase:

[Fully Convolutional Networks for Semantic Segmentation](https://arxiv.org/abs/1605.06211) CVPR 2015

[DeepLab: Semantic Image Segmentation with Deep Convolutional Nets, Atrous Convolution, and Fully Connected CRFs](https://arxiv.org/abs/1606.00915) TPAMI 2017

[Rethinking Atrous Convolution for Semantic Image Segmentation](https://arxiv.org/abs/1706.05587) ArXiv preprint 2017

[ENet: A Deep Neural Network Architecture for Real-Time Semantic Segmentation](https://arxiv.org/abs/1606.02147) ArXiv preprint 2016 

[ERFNet: Efficient Residual Factorized ConvNet for Real-Time Semantic Segmentation](https://ieeexplore.ieee.org/abstract/document/8063438/) ITS 2017

[Spatial As Deep: Spatial CNN for Traffic Scene Understanding](https://arxiv.org/abs/1712.06080) AAAI 2018

[RESA: Recurrent Feature-Shift Aggregator for Lane Detection](https://arxiv.org/abs/2008.13719) AAAI 2021

[Polynomial Regression Network for Variable-Number Lane Detection](http://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123630698.pdf) ECCV 2020

[End-to-end Lane Shape Prediction with Transformers](https://arxiv.org/abs/2011.04233) WACV 2021

You are also welcomed to make additions on this paper list, or open-source your related works here.

## Notes:

1. Cityscapes dataset is down-sampled by 2 when training at 256 x 512, to specify different sizes, modify them in config files if needed.

2. Training times are measured on **a single RTX 2080Ti**, including online validation time for segmentation, test time for lane detection.

3. All segmentation results reported are from single model without CRF and without multi-scale testing.

## Credits:

PytorchAutoDrive is maintained by Zhengyang Feng ([VoldemortX](https://github.com/voldemortX)) and Shaohua Guo ([cedricgsh](https://github.com/cedricgsh)).

Community contributors (GitHub ID): [kalkun](https://github.com/kalkun)

People who sponsored us with hardware: Junshu Tang ([junshutang](https://github.com/junshutang))
