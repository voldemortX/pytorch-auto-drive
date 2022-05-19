# FCN

> [Fully Convolutional Networks for Semantic Segmentation](https://arxiv.org/abs/1605.06211) CVPR 2015

## Method Overview

Fully Convolutional Network (FCN) was the most classic semantic segmentation network, its formulation is even before the ResNet era. Our implementation of FCN borrows from TorchVision, which is a modern take of FCN with ResNet backbone. The "replacing stride by dilation" option from Deeplab is integrated to maintain a 8x down-sampled feature map that makes prediction through a [simple segmentation head](/utils/models/segmentation/fcn.py). It can still be seen as the most simple segmentation baseline, but it is not the original FCN anymore.

<div align=center>
<img src="https://user-images.githubusercontent.com/32259501/158007622-819f07d2-0a4c-4481-9fdf-10da1e5da8d4.png"/>
</div>

## Results

*Training time estimated with single 2080 Ti.*

*ImageNet pre-training, 3-times average/best.*

### PASCAL VOC 2012 trainaug (val)

| backbone | resolution | training time | precision | mIoU (avg) | mIoU | |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| ResNet-101 | 321 x 321 | 3.3h | mix | 70.72 | 70.83 | [model](https://drive.google.com/file/d/1SIIpApBdL0wXanlLeLWBSJJmX3AYLBnf/view?usp=sharing) \| [shell](/tools/shells/fcn_pascalvoc_321x321.sh) |
| ResNet-101 | 321 x 321 | 6.3h | full | 70.91 | 71.55 | [model](https://drive.google.com/file/d/1ZunsGFjXxSIwR8Blckyk-Ils6IdhSqV1/view?usp=sharing) \| [shell](/tools/shells/fcn_pascalvoc_321x321_fp32.sh) |

### Cityscapes (val)

| backbone | resolution | training time | precision | mIoU (avg) | mIoU | |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| ResNet-101 | 321 x 321 | 2.2h | mix | 68.05 | 68.20 | [model](https://drive.google.com/file/d/1zT-lBElfkD1Sratu4WYiTCRU9PF16lLj/view?usp=sharing) \| [shell](/tools/shells/fcn_cityscapes_256x512.sh) |

## Profiling

*FPS is best trial-avg among 3 trials on a 2080 Ti.*

| backbone | resolution | FPS | FLOPS(G) | Params(M) |
| :---: | :---: | :---: | :---: | :---: |
| ResNet-101 | 256 x 512 | 43.32 | 216.42 | 51.95 |
| ResNet-101 | 512 x 1024 | 12.06 | 865.69 | 51.95 |
| ResNet-101 | 1024 x 2048 | 3.06 | 3462.77 | 51.95 |

## Citation

```
@inproceedings{long2015fully,
  title={Fully convolutional networks for semantic segmentation},
  author={Long, Jonathan and Shelhamer, Evan and Darrell, Trevor},
  booktitle={Computer Vision and Pattern Recognition},
  year={2015}
}

@article{shelhamer2016fully,
  title={Fully convolutional networks for semantic segmentation},
  author={Shelhamer, Evan and Long, Jonathan and Darrell, Trevor},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
  volume={39},
  number={4},
  pages={640--651},
  year={2016}
}
```
