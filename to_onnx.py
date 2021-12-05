# Convert only the pt model part

import argparse
import torch
import yaml

from utils.all_utils_landec import build_lane_detection_model as build_lane_model
from utils.all_utils_semseg import build_segmentation_model, load_checkpoint
from tools.onnx_utils import add_basic_arguments, pt_to_onnx, test_conversion, MINIMAL_OPSET_VERSIONS


if __name__ == '__main__':
    # Settings
    parser = argparse.ArgumentParser(description='PyTorch Auto-drive')
    add_basic_arguments(parser)
    args = parser.parse_args()
    with open('configs.yaml', 'r') as f:  # Safer and cleaner than box/EasyDict
        configs = yaml.load(f, Loader=yaml.Loader)
    input_sizes = (args.height, args.width)
    if args.task == 'lane':
        num_classes = configs[configs['LANE_DATASETS'][args.dataset]]['NUM_CLASSES']
        net = build_lane_model(args, num_classes, tracing=True)
        net_without_tracing = build_lane_model(args, num_classes, tracing=False)
    elif args.task == 'seg':
        num_classes = configs[configs['SEGMENTATION_DATASETS'][args.dataset]]['NUM_CLASSES']
        net, _, _, _ = build_segmentation_model(configs, args, num_classes, 0, input_sizes)
        net_without_tracing = net
    else:
        raise ValueError('Task must be lane or seg! Not {}'.format(args.task))

    device = torch.device('cpu')
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    print(device)
    net.to(device)
    net_without_tracing.to(device)
    if args.continue_from is not None:
        load_checkpoint(net=net, optimizer=None, lr_scheduler=None, filename=args.continue_from, strict=False)
        load_checkpoint(net=net_without_tracing, optimizer=None, lr_scheduler=None, filename=args.continue_from)
    else:
        raise ValueError('Must provide a weight file by --continue-from')
    torch.manual_seed(7)
    dummy = torch.randn(1, 3, args.height, args.width, device=device, requires_grad=False)

    # Convert
    onnx_filename = args.continue_from[:args.continue_from.rfind('.')] + '.onnx'
    op_v = 9
    if args.task == 'lane' and args.method in MINIMAL_OPSET_VERSIONS.keys():
        op_v = MINIMAL_OPSET_VERSIONS[args.method]
    if args.task == 'seg' and args.model in MINIMAL_OPSET_VERSIONS.keys():
        op_v = MINIMAL_OPSET_VERSIONS[args.model]
    pt_to_onnx(net, dummy, onnx_filename, opset_version=op_v)

    # Test
    test_conversion(net_without_tracing, onnx_filename, dummy)
