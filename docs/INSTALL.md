# Installation

## Download the code:

```
git clone https://github.com/voldemortX/pytorch-auto-drive.git
cd pytorch-auto-drive
```

## Requirements

- Python >= 3.6
- CUDA >= 9.2 (for CUDA version < 9.2, the code is tested only with PyTorch 1.3 & CUDA 9.0 & CuDNN 7.6.0)
- PyTorch >= 1.6
- TorchVision >= 0.7.0
- [mmcv-full](https://github.com/open-mmlab/mmcv) >= 1.3.5 (according to PyTorch/CUDA version)
- Other pip dependencies: `pip install -r requirements.txt`

The default Conda env (step-by-step):

```
conda create -n pad python=3.6
conda activate pad
conda install pytorch==1.6.0 torchvision==0.7.0 cudatoolkit=10.2 -c pytorch
pip install mmcv-full==1.3.5 -f https://download.openmmlab.com/mmcv/dist/cu102/torch1.6.0/index.html
pip install -r requirements.txt
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
