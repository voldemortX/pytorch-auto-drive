import argparse
import numpy as np
import torch

from utils.tensorrt_utils import build_engine, inference_trt, load_engine
from utils.onnx_utils import inference_onnx


if __name__ == '__main__':
    # Settings
    parser = argparse.ArgumentParser(description='PytorchAutoDrive ONNX to TensorRT')
    parser.add_argument('--onnx-path', type=str, default='', help='ONNX file path')
    parser.add_argument('--height', type=int, default=288,
                        help='Image input height (default: 288)')
    parser.add_argument('--width', type=int, default=800,
                        help='Image input width (default: 800)')
    args = parser.parse_args()

    engine_path = build_engine(args.onnx_path)
    engine = load_engine(engine_path)
    torch.manual_seed(7)
    device = torch.device('cuda:0')
    dummy = torch.randn(1, 3, args.height, args.width, device=device, requires_grad=False)
    with torch.no_grad():
        trt_outputs = inference_trt(engine_binary=engine, input_tensor=dummy)
        onnx_outputs = inference_onnx(dummy.cpu(), args.onnx_path)
        diff = 0.0
        avg = 0.0
        for k, temp_onnx in onnx_outputs.items():
            temp_trt = trt_outputs[k]
            diff += np.abs((temp_onnx - temp_trt.cpu().numpy())).mean()
            avg += np.mean(np.absolute(temp_onnx))
        diff /= len(onnx_outputs)
        avg /= len(onnx_outputs)
        diff_percentage = diff / avg * 100
        print('Average diff: {}\nAverage diff (%): {}'.format(diff, diff_percentage))
        assert diff_percentage < 0.1, 'Diff over 0.1%, please check your environments!'

    print('TensorRT engine saved as: {}'.format(engine_path))
