# Installation

## Requirements

- Python >= 3.6
- CUDA >= 9.2 (for CUDA version < 9.2, the code is tested only with PyTorch 1.3 & CUDA 9.0 & CuDNN 7.6.0)
- PyTorch >= 1.6
- TorchVision >= 0.7.0
- [mmcv-full](https://github.com/open-mmlab/mmcv) (according to PyTorch/CUDA version)

```
pip install -r requirements.txt
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

## Improve training speed with [Pillow-SIMD](https://github.com/uploadcare/pillow-simd) (optional, advanced):

```
pip uninstall pillow
CC="cc -mavx2" pip install -U --force-reinstall pillow-simd
```

Note that you need to use ToTensor transform as late as possible for this speedup.

## Enable tensorboard (optional):

```
tensorboard --logdir=<path to tb_logs>
```

`<path to tb_logs>` is usually `./checkpoints/tb_logs` if you did not customized `save_dir` in config file.
