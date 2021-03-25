import yaml
import argparse

from utils.all_utils_landec import build_lane_detection_model as build_lane_model
from utils.all_utils_semseg import build_segmentation_model
from utils.all_utils_semseg import load_checkpoint
from tools.lane_speed import init as lane_init
from tools.segmentation_speed import init as segmentation_init
from tools.lane_speed import lane_speed_evaluate, lane_speed_evaluate_simple
# from tools.segmentation_speed import segmentation_speed_evaluate_simple, segmentation_speed_evaluate
import torch
import numpy as np

# from torch.cuda.amp import autocast
# from tqdm import tqdm
# from PIL import Image
# from utils.datasets import CULane, StandardLaneDetectionDataset
# from utils.all_utils_semseg import load_checkpoint
# from tools.vis_tools import lane_detection_visualize_batched, simple_lane_detection_transform
# from transforms import functional as F
# from transforms.transforms import ToTensor
# from transforms import ToTensor, Normalize, Resize, RandomRotation, Compose


if __name__ == '__main__':
    # Settings
    parser = argparse.ArgumentParser(description='PyTorch Auto-drive')
    parser.add_argument('--height', type=int, default=288,
                        help='Image input height (default: 288)')
    parser.add_argument('--width', type=int, default=800,
                        help='Image input width (default: 800)')
    parser.add_argument('--dataset', type=str, default='tusimple',
                        help='Profile on TuSimple (tusimple) / CULane (culane) (default: tusimple)')
    parser.add_argument('--method', type=str, default='baseline',
                        help='method selection (lstr/scnn/sad/baseline) (default: baseline)')
    parser.add_argument('--backbone', type=str, default='erfnet',
                        help='backbone selection (erfnet/enet/vgg16/resnet18s/resnet18/resnet34/resnet50/resnet101)'
                             '(default: erfnet)')
    parser.add_argument('--mixed-precision', action='store_true', default=False,
                        help='Enable mixed precision training (default: False)')
    parser.add_argument('--task', type=str, default='lane',
                        help='task selection (lane/segmentation)')
    parser.add_argument('--mode', type=str, default='simple',
                        help='Profiling mode (simple/real)')
    parser.add_argument('--model', type=str, default='deeplabv3',
                        help='Model selection (fcn/erfnet/deeplabv2/deeplabv3/enet) (default: deeplabv3)')
    parser.add_argument('--continue-from', type=str, default=None,
                        help='Continue training from a previous checkpoint')
    args = parser.parse_args()

    lane_need_interpolate = ['vgg16', 'resnet18s', 'resnet18', 'resnet34', 'resnet50', 'resnet101']
    seg_need_interpolate = ['fcn', 'deeplabv2', 'deeplabv3']
    with open('configs.yaml', 'r') as f:  # Safer and cleaner than box/EasyDict
        configs = yaml.load(f, Loader=yaml.Loader)
    # if args.dataset not in configs['LANE_DATASETS'].keys():
    #     raise ValueError

    mean = configs['GENERAL']['MEAN']
    std = configs['GENERAL']['STD']

    if args.task == 'lane':

        num_classes = configs[configs['LANE_DATASETS'][args.dataset]]['NUM_CLASSES']
        count_interpolate = False
        if args.backbone in lane_need_interpolate:
            count_interpolate = True
        if torch.cuda.is_available():
            device = torch.device('cuda:0')
        net = build_lane_model(args, num_classes)
        print(device)
        net.to(device)
        load_checkpoint(net=net, optimizer=None, lr_scheduler=None, filename=args.continue_from)

        print('Profiling, please clear your GPU memory before doing this.')
        if args.mode == 'simple':
            dummy = torch.ones((1, 3, args.height, args.width))

            fps = lane_speed_evaluate_simple(net=net, device=device, dummy=dummy, num=300, count_interpolate=True)
            print("GPU FPS: " + str(fps))
        elif args.mode == 'real' and args.dataset in configs['LANE_DATASETS'].keys():
            base = configs[configs['LANE_DATASETS'][args.dataset]]['BASE_DIR']
            val_loader = lane_init(dataset=args.dataset, input_sizes=(args.height, args.width), mean=mean, std=std,
                                   base=base)
            fps, gpu_fps = lane_speed_evaluate(net=net, device=device, loader=val_loader, num=300,
                                               count_interpolate=True)
            print("Real FPS: " + str(fps))
            print("GPU FPS: " + str(gpu_fps))
        else:
            raise ValueError

    elif args.task == 'segmentation':
        num_classes = configs[configs['SEGMENTATION_DATASETS'][args.dataset]]['NUM_CLASSES']
        input_sizes = (args.height, args.width)

        count_interpolate = False
        if args.backbone in seg_need_interpolate:
            count_interpolate = True
        if torch.cuda.is_available():
            device = torch.device('cuda:0')
        print(device)
        city_aug = 0
        net, city_aug, _, _ = build_segmentation_model(configs, args, num_classes, city_aug, input_sizes)
        net.to(device)
        # load_checkpoint(net=net, optimizer=None, lr_scheduler=None, filename=args.continue_from)

        print('Profiling, please clear your GPU memory before doing this.')
        if args.mode == 'simple':
            dummy = torch.ones((1, 3, args.height, args.width))
            fps = lane_speed_evaluate_simple(net=net, device=device, dummy=dummy, num=300,
                                             count_interpolate=True)
            print("GPU FPS: " + str(fps))
        elif args.mode == 'real' and args.dataset in configs['SEGMENTATION_DATASETS'].keys():
            base = configs[configs['SEGMENTATION_DATASETS'][args.dataset]]['BASE_DIR']
            train_label_id_map = configs[configs['SEGMENTATION_DATASETS'][args.dataset]]['LABEL_ID_MAP'] if \
                'LABEL_ID_MAP' in configs[configs['SEGMENTATION_DATASETS'][args.dataset]].keys() else \
                configs['CITYSCAPES']['LABEL_ID_MAP']

            val_loader = segmentation_init(dataset=args.dataset, input_sizes=(args.height, args.width), mean=mean,
                                           std=std, test_base=base, city_aug=city_aug, train_base=None,
                                           train_label_id_map=None, test_label_id_map=train_label_id_map)
            fps, gpu_fps = lane_speed_evaluate(net=net, device=device, loader=val_loader, num=300,
                                               count_interpolate=True)
            print("Real FPS: " + str(fps))
            print("GPU FPS: " + str(gpu_fps))
        else:
            raise ValueError
