# My segmentation codebase
Segmentation models (DeeplabV3, DeeplabV2, etc.) based on Python 3.6.8 and 

PyTorch Pytorch 1.2.0 (cuda 10.0) & torchvision 0.4.0 with mixed precision training, since 1.2.0 is 100% compatible with apex

Including modulated (borrowed) mIOU & pixel acc calculation, "poly" learning rate schedule, basic input transformations and visulizations, also tests of mixed precision training

### Currently supported datasets: 

PASCAL VOC 2012 (Deeplab 10582 trainaug version, I don't think I have the right to distribute this dataset, so just get the images yourself); Cityscapes

### Currently supported models:

DeeplabV3, DeeplabV2, Also you can use PSPNet and FCN models in torchvision

## Usage(Linux):

Setup apex with a python3 enviroment (cuda 10):

```
pip install https://download.pytorch.org/whl/cu100/torch-1.2.0-cp36-cp36m-manylinux1_x86_64.whl && pip install https://download.pytorch.org/whl/cu100/torchvision-0.4.0-cp36-cp36m-manylinux1_x86_64.whl
git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
```

Download the code:

```
git clone https://github.com/voldemortX/pytorch-segmentation.git
cd code
```

Prepare the code:

1. Change the 2 base directories in code/data_processing.py
2. Run cityscapes_data_list.py

Enable tensorboard:

```
tensorboard --logdir=runs
```

Run mixed-precision training on PASCAL VOC 2012 with DeeplabV2 (73.5% mIOU, averaged across 3 runs, see tensorboard logs in code/runs/):

```
python main.py --epochs=30 --lr=0.002 --batch-size=8 --dataset=voc --model=deeplabv2 --mixed-precision
```

Other commands, e.g. run full-precision training on Cityscapes with DeeplabV3:

```
python main.py --epochs=60 --lr=0.002 --batch-size=8 --dataset=city --model=deeplabv3
```

## Notes:

Experiments used same random seeds. However, it is still not deterministic due to parallel computing and other unknown factors.

Cityscapes dataset is down-sampled by 2, to specify different sizes, modify this [line](code/data_processing.py#L32); similar changes can be down with PASCAL VOC 2012.

On **a single RTX 2080Ti**, training DeeplabV3 (30 epochs at 321x321 resolution) takes **~9h15m** and **~8.5G** GPU memory (or **~6h35m** and **~5.5G** GPU memory with mixed precision training)
