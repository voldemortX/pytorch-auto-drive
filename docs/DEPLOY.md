# Deployment Guide

- [x] PyTorch -> ONNX
- [ ] ONNX -> TensorRT
- [ ] ONNX inference
- [ ] TensorRT inference
- [ ] ONNX visualization
- [ ] TensorRT visualization

## Installation

Install all deployment packages by:

Or you can install only what is needed in each part of this tutorial, referring to **Extra Dependencies**.

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
