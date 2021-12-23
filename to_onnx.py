# Convert only the pt model part
import argparse
import os
import torch

from utils.common import load_checkpoint
from utils.args import read_config, parse_arg_cfg
from utils.models import MODELS
from tools.onnx_utils import pt_to_onnx, test_conversion, get_minimal_opset_version, append_trace_arg


if __name__ == '__main__':
    # Settings
    parser = argparse.ArgumentParser(description='PyTorch Auto-drive')
    parser.add_argument('--config', type=str, help='Path to config file', required=True)
    parser.add_argument('--height', type=int, default=288,
                        help='Image input height (default: 288)')
    parser.add_argument('--width', type=int, default=800,
                        help='Image input width (default: 800)')

    # Optional args/to overwrite configs
    group2 = parser.add_mutually_exclusive_group()
    group2.add_argument('--continue-from', type=str,
                        help='[Deprecated] Continue training from a previous checkpoint')
    group2.add_argument('--checkpoint', type=str,
                        help='Continue/Load from a previous checkpoint')

    args = parser.parse_args()

    defaults = {
        'checkpoint': None,
        'continue_from': None
    }

    deprecation_map = {
        'continue_from': {'valid': 'checkpoint', 'message': ''}
    }

    # Parse configs and build model
    cfg = read_config(args.config)
    args, _ = parse_arg_cfg(args, cfg['test'], defaults, required=None, deprecation_map=deprecation_map)
    net_without_tracing = MODELS.from_dict(cfg['model'])
    trace_arg = {
        'h': args.height,
        'w': args.width,
        'bs': 1
    }
    cfg_with_trace_arg = append_trace_arg(cfg['model'].copy(), trace_arg)
    net = MODELS.from_dict(cfg_with_trace_arg)

    # Move to device (simple single card)
    device = torch.device('cpu')
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    print(device)
    net.to(device)
    net_without_tracing.to(device)

    # Load weights
    if args.checkpoint is not None:
        load_checkpoint(net=net, optimizer=None, lr_scheduler=None, filename=args.checkpoint, strict=False)
        load_checkpoint(net=net_without_tracing, optimizer=None, lr_scheduler=None, filename=args.checkpoint)
    else:
        raise ValueError('Must provide a weight file by --checkpoint')

    # Set dummy for precision matching
    torch.manual_seed(7)
    dummy = torch.randn(1, 3, args.height, args.width, device=device, requires_grad=False)

    # Convert
    onnx_filename = args.checkpoint[:args.checkpoint.rfind('.')] + '.onnx'
    op_v = get_minimal_opset_version(cfg['model'], -1)
    print('Minimum required opset version is: {}'.format(op_v))
    pt_to_onnx(net, dummy, onnx_filename, opset_version=op_v)

    # Test
    test_conversion(net_without_tracing, onnx_filename, dummy)

    print('ONNX model saved as: {}'.format(onnx_filename))
