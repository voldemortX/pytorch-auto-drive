# Installation

## Requirements

- Python >= 3.6
- CUDA >= 9.2 (CUDA 10 is preferred, for CUDA version < 9.2, refer to [cu90torch13](https://github.com/voldemortX/pytorch-auto-drive/tree/cu90torch13) branch: PyTorch 1.3 & CUDA 9.0 & CuDNN 7.6.0)
- PyTorch >= 1.6 
- TorchVision >= 0.7.0
- [mmcv-full](https://github.com/open-mmlab/mmcv) (according to PyTorch/CUDA version)

```
pip install tqdm ujson tensorboard imageio opencv-python pyyaml thop Shapely p-tqdm scipy sklearn filetype
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
