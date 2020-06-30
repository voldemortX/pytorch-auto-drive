# My segmentation codebase
Segmentation models (Deeplab, FCN, ERFNet) based on Python 3.6.8 and 

Pytorch 1.2.0 (cuda 10.0) & torchvision 0.4.0 with mixed precision training, since 1.2.0 is 100% compatible with apex.

Including modulated (borrowed) mIOU & pixel acc calculation, "poly" learning rate schedule, basic input transformations and visulizations, also tests of mixed precision training.

### Currently supported datasets: 

PASCAL VOC 2012 (Deeplab 10582 *trainaug* version, I don't think I have the right to distribute this dataset, so just get the images yourself).

Cityscapes.

TuSimple (in progress).

CULane (in progress).

### Currently supported models:

ResNet-101 backbone: DeeplabV3, DeeplabV2, FCN

Specialized real-time backbone: ERFNet

(You can of course also use other backbones in torchvision by simply calling a different function or other models by using the most recent torchvision implementation)

### Performance (ImageNet pre-training, averaged across 3 runs)

| model | mixed precision? | Dataset | mIoU (%) |
| :---: | :---: | :---: | :---: |
| FCN | *yes* | PASCAL VOC 2012 | 69.09 |
| FCN | *no* | PASCAL VOC 2012 | 69.16 |
| DeepLabV2 | *yes* | PASCAL VOC 2012 | 72.88 |
| DeepLabV3 | *yes* | PASCAL VOC 2012 | 76.70 |
| FCN | *yes* | Cityscapes | 65.57 |
| DeepLabV2 | *yes* | Cityscapes | 66.72 |
| DeepLabV3 | *yes* | Cityscapes | 67.84 |
| DeepLabV3 | *no* | Cityscapes | 67.76 |
| ERFNet| *no* | Cityscapes | |

*Note that the best run from ERFNet is 72.1% in mIoU, same as the original implementation by the authors.*

## Usage(Linux):

Setup apex with a python3 enviroment (cuda 10):

```
pip install https://download.pytorch.org/whl/cu100/torch-1.2.0-cp36-cp36m-manylinux1_x86_64.whl && pip install https://download.pytorch.org/whl/cu100/torchvision-0.4.0-cp36-cp36m-manylinux1_x86_64.whl
git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
```

!There seems to be an issue of apex installations from the official repo sometimes. If you encounter errors, we suggest you use our stored older apex [codes](https://drive.google.com/open?id=1x8enpvdTTZ3RChf17XvcLdSYulUPg3sR).

Download the code:

```
git clone https://github.com/voldemortX/pytorch-segmentation.git
cd code
```

Prepare the code:

1. Change the 2 base directories in code/data_processing.py
2. Run cityscapes_data_list.py
3. If you are using ERFNet, remember to download the ImageNet pre-trained weights *erfnet_encoder_pretrained.pth.tar* from [here](https://github.com/Eromera/erfnet_pytorch/tree/master/trained_models), and put it in the **code** folder

Enable tensorboard:

```
tensorboard --logdir=runs
```

Run mixed-precision training on PASCAL VOC 2012 with DeeplabV2:

```
python main.py --epochs=30 --lr=0.002 --batch-size=8 --dataset=voc --model=deeplabv2 --mixed-precision
```

Other commands, e.g. run full-precision training on Cityscapes with DeeplabV3:

```
python main.py --epochs=60 --lr=0.002 --batch-size=8 --dataset=city --model=deeplabv3
```

Or run full-precision training on Cityscapes with ERFNet:

```
python main.py --epochs=150 --lr=0.0005 --batch-size=6 --dataset=city --model=ERFNet --val-num-steps=500
```

## Notes:

Note that ERFNet currently does not support mixed precision training, due to a bug in PyTorch 1.2.0. It will be supported when this repo is updated to PyTorch 1.6, which will happen soon.

Most experiments used same random seeds. However, it is still not deterministic due to parallel computing and other unknown factors.

Cityscapes dataset is down-sampled by 2, to specify different sizes, modify this [line](code/data_processing.py#L32); similar changes can be down with PASCAL VOC 2012.

On **a single RTX 2080Ti**, training DeeplabV3 (30 epochs at 321x321 resolution) takes **~9h15m** and **~8.5G** GPU memory (or **~6h35m** and **~5.5G** GPU memory with mixed precision training).
