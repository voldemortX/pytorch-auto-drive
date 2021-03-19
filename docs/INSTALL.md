# Installation

## Requirements

- Python >= 3.6
- CUDA 10
- PyTorch >= 1.6 
- TorchVision >= 0.7.0
- tqdm  `pip install tqdm`
- ujson  `pip install ujson`
- tensorboard  `pip install tensorboard`
- numpy  `will be installed along with PyTorch`
- imageio  `pip install imageio`
- OpenCV  `pip install opencv-python`
- pillow  `will be installed along with TorchVision`
- Pyyaml  `pip install pyyaml`

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
