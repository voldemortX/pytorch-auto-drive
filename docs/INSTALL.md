# Installation

## Requirements
- Python >= 3.6
- CUDA 10
- PyTorch >= 1.6 
- TorchVision >= 0.7.0
- tqdm
- ujson
- tensorboard
- numpy
- imageio
- opencv-python
- Pillow
- pyyaml.

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

## Enable tensorboard (optional):

```
tensorboard --logdir=runs
```