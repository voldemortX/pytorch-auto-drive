# Installation

## Requirements

- Python >= 3.6
- CUDA 10
- PyTorch >= 1.6 
- TorchVision >= 0.7.0

```
pip install tqdm ujson tensorboard imageio opencv-python pyyaml thop Shapely p-tqdm scipy sklearn
```

## Download the code:
   
```
git clone https://github.com/voldemortX/pytorch-auto-drive.git
cd pytorch-auto-drive
```

## Prepare the code:

```
chmod 777 *.sh tools/shells/*.sh
mkdir output
```

## Improve training speed with [Pillow-SIMD](https://github.com/uploadcare/pillow-simd) (optional):

```
pip uninstall pillow
CC="cc -mavx2" pip install -U --force-reinstall pillow-simd
```

## Enable tensorboard (optional):

```
tensorboard --logdir=runs
```
