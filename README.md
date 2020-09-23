# Codebase for deep autonomous driving perception tasks

Segmentation models (Deeplab, FCN, ERFNet), Lane detection models (in progress) based on Python 3.6 and Pytorch 1.6.0 (cuda 10.0) & torchvision 0.7.0 with mixed precision training.

Including modulated (borrowed) mIOU & pixel acc calculation, "poly" learning rate schedule, basic input loading, transformations and visulizations, also tests of mixed precision training and tensorboard logging.

## Currently supported datasets: 

PASCAL VOC 2012 (10582 *trainaug* version).

Cityscapes.

GTAV (The UDA baseline setup: GTAV 24966 training set, with cityscapes *val* set validation).

TuSimple (In progress).

CULane (In progress).

## Currently supported models:

ResNet-101 backbone: DeeplabV3, DeeplabV2, FCN

Specialized real-time backbone: ERFNet

*You can of course also use other backbones (e.g. ResNet-50) in torchvision by simply calling a different function by using the most recent torchvision implementation*

## Performance (ImageNet pre-training, val accuracy averaged across 3 runs):

| model | resolution | mixed precision? | Dataset | mIoU (%) | Training time |
| :---: | :---: | :---: | :---: | :---: | :---: |
| FCN | 321 x 321 | *yes* | PASCAL VOC 2012 | 69.46 | 3.3h |
| FCN | 321 x 321 | *no* | PASCAL VOC 2012 | 69.16 | |
| DeepLabV2 | 321 x 321 | *yes* | PASCAL VOC 2012 | 72.84 | 3.3h |
| DeepLabV3 | 321 x 321 | *yes* | PASCAL VOC 2012 | 77.05 | 6.9h |
| FCN | 257 x 513 | *yes* | Cityscapes | 65.79 | 2.3h |
| DeepLabV2 | 257 x 513 | *yes* | Cityscapes | 66.89 | 2.3h |
| DeepLabV3 | 257 x 513 | *yes* | Cityscapes | 67.87 | 4.8h |
| DeepLabV3 | 257 x 513 | *no* | Cityscapes | 67.76 | |
| ERFNet| 512 x 1024 | *no* | Cityscapes | 71.99 | 5h |

*\*Note that the best run from ERFNet on Cityscapes val is 72.47% in mIoU, slightly better than the original implementation by the authors (72.2%).*

## Usage(Linux):

### Prepare the datasets:

The PASCAL VOC 2012 dataset we used is the commonly used 10582 training set version. If you don't already have that dataset, we refer you to [Google](https://www.google.com) or this [blog](https://www.sun11.me/blog/2018/how-to-use-10582-trainaug-images-on-DeeplabV3-code/). Other datasets can be downloaded in their official websites.

Setup a Python3 enviroment (cuda 10), with PyTorch 1.6, TorchVision 0.7.0, tqdm, tensorboard, numpy.

### Download the code:

```
git clone https://github.com/voldemortX/pytorch-segmentation.git
cd code
```

### Prepare the code:

1. Change the 2 base directories in code/data_processing.py
2. Run cityscapes_data_list.py
3. If you are using ERFNet, remember to download the ImageNet pre-trained weights *erfnet_encoder_pretrained.pth.tar* from [here](https://github.com/Eromera/erfnet_pytorch/tree/master/trained_models), and put it in the **code** folder

### Enable tensorboard:

```
tensorboard --logdir=runs
```

### Run the code:

Run mixed precision training on PASCAL VOC 2012 with DeeplabV2:

```
python main_semseg.py --epochs=30 --lr=0.002 --batch-size=8 --dataset=voc --model=deeplabv2 --mixed-precision
```

Other commands, e.g. run full precision training on Cityscapes with DeeplabV3:

```
python main_semseg.py --epochs=60 --lr=0.002 --batch-size=8 --dataset=city --model=deeplabv3
```

Or run mixed precision training on Cityscapes with ERFNet:

```
python main_semseg.py --epochs=150 --lr=0.0007 --batch-size=10 --dataset=city --model=erfnet --mixed-precision
```

## Notes:

Cityscapes dataset is down-sampled by 2, to specify different sizes, modify this [line](code/data_processing.py#L32); similar changes can be done with PASCAL VOC 2012.

Training times are measured on **a single RTX 2080Ti**.
