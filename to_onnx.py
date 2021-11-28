# Convert only the pt model part

import argparse
import torch
import yaml
import fcntl

from utils.all_utils_landec import build_lane_detection_model as build_lane_model
from utils.all_utils_landec import init, test_one_set, fast_evaluate
from utils.all_utils_semseg import build_segmentation_model, load_checkpoint
from tools.onnx_utils import add_basic_arguments, pt_to_onnx, test_conversion, MINIMAL_OPSET_VERSIONS, get_ort_session


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
        net = build_lane_model(args, num_classes)
    elif args.task == 'seg':
        num_classes = configs[configs['SEGMENTATION_DATASETS'][args.dataset]]['NUM_CLASSES']
        net, _, _, _ = build_segmentation_model(configs, args, num_classes, 0, input_sizes)
    else:
        raise ValueError('Task must be lane or seg! Not {}'.format(args.task))

    device = torch.device('cpu')
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    print(device)
    net.to(device)
    if args.continue_from is not None:
        load_checkpoint(net=net, optimizer=None, lr_scheduler=None, filename=args.continue_from)
    else:
        raise ValueError('Must provide a weight file by --continue-from')
    torch.manual_seed(7)
    mean = configs['GENERAL']['MEAN']
    std = configs['GENERAL']['STD']
    if args.dataset not in configs['LANE_DATASETS'].keys():
        raise ValueError

    # temp variable for inference
    real_height, real_width = input_sizes[0]
    dummy = torch.randn(1, 3, real_height, real_width, device=device, requires_grad=False)
    # Convert
    onnx_filename = args.continue_from[:args.continue_from.rfind('.')] + '.onnx'
    op_v = 9
    if args.task == 'lane' and args.method in MINIMAL_OPSET_VERSIONS.keys():
        op_v = MINIMAL_OPSET_VERSIONS[args.method]
    if args.task == 'seg' and args.model in MINIMAL_OPSET_VERSIONS.keys():
        op_v = MINIMAL_OPSET_VERSIONS[args.model]
    # TODO: directly load xxx.onnx without converting
    pt_to_onnx(net, dummy, onnx_filename, opset_version=op_v)

    if args.verify == 'no':
        print("The model has been converted.")
    elif args.verify == 'simple':
        test_conversion(net, onnx_filename, dummy)
    elif args.verify == 'real':
        num_classes = configs[configs['LANE_DATASETS'][args.dataset]]['NUM_CLASSES']
        input_sizes = configs[configs['LANE_DATASETS'][args.dataset]]['SIZES']
        gap = configs[configs['LANE_DATASETS'][args.dataset]]['GAP']
        ppl = configs[configs['LANE_DATASETS'][args.dataset]]['PPL']
        thresh = configs[configs['LANE_DATASETS'][args.dataset]]['THRESHOLD']
        weights = configs[configs['LANE_DATASETS'][args.dataset]]['WEIGHTS']
        base = configs[configs['LANE_DATASETS'][args.dataset]]['BASE_DIR']
        max_lane = configs[configs['LANE_DATASETS'][args.dataset]]['MAX_LANE']
        ort_net = get_ort_session(onnx_filename)

        # onnx inference
        if args.state == 1 or args.state == 2 or args.state == 3:
            data_loader = init(batch_size=args.batch_size, state=args.state, dataset=args.dataset,
                               input_sizes=input_sizes, mean=mean, std=std, base=base, workers=args.workers,
                               method=args.method)
            load_checkpoint(net=net, optimizer=None, lr_scheduler=None, filename=args.continue_from)
            if args.state == 1:  # Validate with mean IoU
                _, x = fast_evaluate(loader=data_loader, device=device, net=ort_net,
                                     num_classes=num_classes, output_size=input_sizes[0],
                                     is_mixed_precision=args.mixed_precision)
                with open('log.txt', 'a') as f:
                    # Safe writing with locks
                    fcntl.flock(f, fcntl.LOCK_EX)
                    f.write(args.exp_name + ' validation: ' + str(x) + '\n')
                    fcntl.flock(f, fcntl.LOCK_UN)
            else:  # Test with official scripts later (so just predict lanes here)
                test_one_set(net=ort_net, device=device, loader=data_loader, is_mixed_precision=args.mixed_precision,
                             gap=gap, input_sizes=input_sizes, ppl=ppl, thresh=thresh, dataset=args.dataset,
                             method=args.method, max_lane=max_lane, exp_name=args.exp_name, deploy='onnx')
    else:
        raise ValueError

# python to_onnx.py --state=2 --continue-from=vgg16_baseline_tusimple_20210223.pt --dataset=tusimple --method=baseline --backbone=vgg16 --batch-size=1 --mixed-precision --task=lane --exp-name=none_onnx_test --verify=real