# Deployment Guide

- [x] PyTorch -> ONNX
- [x] ONNX -> TensorRT
- [ ] ONNX inference
- [ ] TensorRT inference
- [ ] ONNX visualization
- [ ] TensorRT visualization

## Installation

Install all deployment packages (our tested conda version) by:

```
conda install cudatoolkit=10.2 -c pytorch
conda install cudnn==8.0.4 -c nvidia
pip install onnx==1.10.2 onnxruntime-gpu==1.6.0
python3 -m pip install --upgrade nvidia-tensorrt==8.2.1.8
```

In this version, TensorRT may use CUDA runtime >= 11, you might avoid using conda if you have sudo access on your device.

Or you can incrementally install **Extra Dependencies** through the tutorial.

## PyTorch -> ONNX:

**PyTorch version >= 1.6.0 is recommended for this feature.**

### Extra Dependencies:

```
pip install onnx==1.10.2 onnxruntime-gpu==<version>
```

`<version>` depends on your CUDA/CuDNN version, see [here](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html#requirements), if you met version issues,
try install exact cudatoolkit and cudnn from conda. Or you can just install the CPU onnxruntime for this functionality.

### Conversion:

The conversion is based on Torch JIT's tracing on random input. To convert a checkpoint (*e.g.,* ckpt.pt) to ONNX, simply run this command:

```
python to_onnx.py --height=<input height> --width=<input width> --task=<task: lane/seg> --dataset=<dataset> --method=<method for lane det> --backbone=<backbone for lane det> --model=<model for seg> --continue-from=ckpt.pt
```

You'll then see the saved `ckpt.onnx` file and a report on the conversion quality.

### Currently Unsupported Models:
- ENet (segmentation)
- ENet backbone (lane detection)

## ONNX -> TensorRT:

### Extra Dependencies:

```
python3 -m pip install --upgrade nvidia-tensorrt==<version>
```

TensorRT `<version>` is recommended to be at least 7.2, you can also install it via other means than pip.
To work better with onnxruntime (for checking of conversion quality), you best checkout the [compatibility](https://onnxruntime.ai/docs/execution-providers/TensorRT-ExecutionProvider.html#requirements).

### Conversion:

The conversion is mainly a building of TensorRT engine. To convert a checkpoint (*e.g.,* ckpt.onnx) to TensorRT, simply run this command:

```
python to_tensorrt.py --height=<input height> --width=<input width> --onnx-path=<ckpt.onnx>
```

You'll then see the saved `ckpt.engine` file and a report on the conversion quality.

### Currently Unsupported Models:
- ENet (segmentation)
- ENet backbone (lane detection)
