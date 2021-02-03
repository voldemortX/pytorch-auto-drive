# Codebase for deep autonomous driving perception tasks

Segmentation models (**Deeplab, FCN, ERFNet**), Lane detection models (**ERFNet, ERFNet-SCNN and others**) based on Python 3.6 and PyTorch >=1.6.0 (CUDA 10) & TorchVision >=0.7.0 with mixed precision training.

**This repository is under active development, which means performance reported could improve in the future.**

## Highlights

Fast probability map to poly line conversion, modulated (borrowed) mIOU & pixel acc calculation, "poly" learning rate schedule, basic input loading, transformations and visualizations, also tests of mixed precision training and tensorboard logging.

And models from this repo is faster (also better or at least similar) than the official implementations and other implementations out there.

## Supported datasets: 

| Task | Dataset |
| :---: | :---: |
| semantic segmentation | PASCAL VOC 2012 |
| semantic segmentation | Cityscapes |
| semantic segmentation | GTAV* |
| semantic segmentation | SYNTHIA* |
| lane detection | CULane |
| lane detection | TuSimple |

\* The UDA baseline setup, with Cityscapes *val* set as validation.

## Supported models:

| Task | Backbone | Model |
| :---: | :---: | :---: |
| semantic segmentation | ResNet-101 | FCN |
| semantic segmentation | ResNet-101 | DeeplabV2 |
| semantic segmentation | ResNet-101 | DeeplabV3 |
| semantic segmentation | - | ERFNet |
| lane detection | ERFNet | ERFNet |
| lane detection | ERFNet | SCNN |
| lane detection | VGG | SCNN |
| lane detection | ERFNet | SAD (*In progress*) |
| lane detection | ERFNet | PRNet (*In progress*) |
| lane detection | ERFNet | LSTR (*In progress*) |
| lane detection | ERFNet | RESA (*In progress*) |
| lane detection | VGG | RESA (*In progress*) |

*You can of course also use other backbones (e.g. ResNet-50) in TorchVision by simply calling a different function in the most recent TorchVision implementation.*

*The VGG backbone for SCNN/RESA/etc. should technically be DeepLab-LargeFOV, we keep calling it VGG for consistency with common practices.*

## Semantic segmentation performance:

| model | resolution | mixed precision? | dataset | mIoU (%) | training time |
| :---: | :---: | :---: | :---: | :---: | :---: |
| FCN | 321 x 321 | *yes* | PASCAL VOC 2012 | 70.72 | 3.3h |
| FCN | 321 x 321 | *no* | PASCAL VOC 2012 | 70.90 | 6.3h |
| DeeplabV2 | 321 x 321 | *yes* | PASCAL VOC 2012 | 74.59 | 3.3h |
| DeeplabV3 | 321 x 321 | *yes* | PASCAL VOC 2012 | 78.11 | 7h |
| FCN | 256 x 512 | *yes* | Cityscapes | 68.05 | 2.2h |
| DeeplabV2 | 256 x 512 | *yes* | Cityscapes | 68.65 | 2.2h |
| DeeplabV3 | 256 x 512 | *yes* | Cityscapes | 69.87 | 4.5h |
| DeeplabV2 | 256 x 512 | *no* | Cityscapes | 68.45 | 4h |
| ERFNet| 512 x 1024 | *yes* | Cityscapes | 71.99 | 5h |
| DeeplabV2 | 512 x 1024 | *yes* | Cityscapes | 71.78 | 9h |
| DeeplabV2 | 512 x 1024 | *yes* | GTAV | 32.90 | 13.8h |
| DeeplabV2 | 512 x 1024 | *yes* | SYNTHIA | 33.89 (mIoU-16) | 10.4h |

*\* All performance is measured with ImageNet pre-training and reported as 3 times average on val set. Note that the best run from ERFNet on Cityscapes val is 72.47% in mIoU, slightly better than the original implementation by the authors (72.2%).*

## Lane detection performance:

| method | backbone | resolution | mixed precision? | dataset | metric | average | best | training time |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Baseline | ERFNet | 288 x 800 | *yes* | CULane | F measure | 73.37 | 73.54 | 6h |
| SCNN | ERFNet | 288 x 800 | *yes* | CULane | F measure | 74.41 | 74.44 | 11.3h |
| Baseline | ERFNet | 360 x 640 | *yes* | TuSimple | Accuracy | 95.15% | 95.24% | 0.8h |
| SCNN | ERFNet | 360 x 640 | *yes* | TuSimple | Accuracy | 96.00% | 96.12% | 1.6h |
| Baseline | VGG | 288 x 800 | *yes* | CULane | F measure |  |  |  |
| SCNN | VGG | 288 x 800 | *yes* | CULane | F measure |  |  |  |

*\* All performance is measured with ImageNet pre-training and reported as 3 times average/best on test set.*

### Tusimple detailed performance (best):

| method | backbone | accuracy | FP | FN |
| :---: | :---: | :---: | :---: | :---: |
| Baseline | ERFNet | 95.24% | 0.0569 | 0.0457 |
| SCNN | ERFNet | 96.12% | 0.0468 | 0.0335 |

### CULane detailed performance (best):

| category | ERFNet-Baseline | ERFNet-SCNN |
| :---: | :---: | :---: |
| normal | 91.35 | 91.84 |
| crowded | 71.45 | 72.55 |
| night | 68.77 | 69.42 |
| no line | 46.05 | 47.99 |
| shadow | 69.21 | 75.59 |
| arrow | 87.19 | 87.32 |
| dazzle light | 62.34 | 63.93 |
| curve | 67.27 | 69.81 |
| crossroad | 1837 | 2160 |
| total | 73.54 | 74.44 |

## Preparations:

1. Setup a Python3 environment (CUDA 10), with PyTorch >= 1.6, TorchVision >= 0.7.0, tqdm, tensorboard, numpy, imageio, opencv-python, yaml.

2. Download the code:
   
```
git clone https://github.com/voldemortX/pytorch-auto-drive.git
cd pytorch-auto-drive
```

3. Prepare the code:

```
chmod 777 *.sh
mkdir output
```

## Enable tensorboard (optional):

```
tensorboard --logdir=runs
```

## Getting started

Get started with [SEGMENTATION.md](SEGMENTATION.md) for semantic segmentation.

Get started with [LANEDETECTION.md](LANEDETECTION.md) for lane detection.

## Notes:

1. Cityscapes dataset is down-sampled by 2 when training at 256 x 512, to specify different sizes, modify them in [configs.yaml](configs.yaml); similar changes can be done with other experiments.

2. Training times are measured on **a single RTX 2080Ti**, including online validation time for segmentation, test time for lane detection.

3. All segmentation results reported are from single model without CRF and without multi-scale testing.

4. PR and issues are always welcomed.
