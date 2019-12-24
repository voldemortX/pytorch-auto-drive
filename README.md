# My segmentation codebase
Segmentation models(Start with DeeplabV3) based on Python 3.6.8 and 

PyTorch 1.3.1 / Pytorch 1.2.0(cuda 10.0)&torchvision 0.4.0 with mixed precision training, as 1.2.0 is 100% compatible with apex

Including modulated(borrowed) mIOU&pixel acc calculation, "poly" learning rate schedule, basic input transformations and visulizations, also tests of mixed precision training

Currently supported datasets: 

PASCAL VOC 2012(Deeplab 10582 trainaug version, I don't think I have the right to distribute this dataset, so just get the images yourself)

Currently supported models:

DeeplabV3(ImageNet pretrained ResNet 101) from torchvision

## Usage(Linux):

Setup apex with a python3 enviroment(cuda 10):

```
pip install https://download.pytorch.org/whl/cu100/torch-1.2.0-cp36-cp36m-manylinux1_x86_64.whl && pip install https://download.pytorch.org/whl/cu100/torchvision-0.4.0-cp36-cp36m-manylinux1_x86_64.whl
git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
```

Download the code:

```
git clone https://github.com/voldemortX/DeeplabV3_PyTorch1.3_Codebase
cd code
```

Enable tensorboard:

```
tensorboard --logdir=runs
```

Run normal training on PASCAL VOC(76.47% mIOU, averaged across 3 runs, see tensorboard logs in code/runs/):

```
python main.py --epochs=30 --lr=0.002 --batch-size=8
```

Run mixed precision training on PASCAL VOC(76.45% mIOU, averaged across 3 runs, see tensorboard logs in code/runs/):

```
python main.py --epochs=30 --lr=0.002 --batch-size=8 --mixed-precision
```

## Notes:

Experiments used same random seeds.However it is still not deterministic due to parallel computing and other things.

On a single RTX 2080Ti, training(30 epochs at 321x321 resolution) takes **~9h15m** and **~8.5G** GPU memory(or **~6h35m** and **~5.5G** GPU memory with mixed precision training)
