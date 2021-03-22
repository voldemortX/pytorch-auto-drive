import yaml
import argparse
from utils.all_utils_landec import build_lane_detection_model as build_lane_model
from utils.all_utils_semseg import load_checkpoint
from tools.lane_speed import init as lane_init
from tools.lane_speed import lane_speed_evaluate
import torch

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
                        help='Train/Evaluate on TuSimple (tusimple) / CULane (culane) (default: tusimple)')
    parser.add_argument('--method', type=str, default='baseline',
                        help='method selection (lstr/scnn/sad/baseline/etc) (default: baseline)')
    parser.add_argument('--backbone', type=str, default='erfnet',
                        help='backbone selection (erfnet/vgg16/resnet18s/resnet18/resnet34/resnet50/resnet101)'
                             '(default: erfnet)')
    parser.add_argument('--mixed-precision', action='store_true', default=False,
                        help='Enable mixed precision training (default: False)')
    parser.add_argument('--task', type=str, default='lane',
                        help='task selection(lane/segmentation')
    parser.add_argument('--continue-from', type=str, default=None,
                        help='Continue training from a previous checkpoint')
    args = parser.parse_args()
    with open('configs.yaml', 'r') as f:  # Safer and cleaner than box/EasyDict
        configs = yaml.load(f, Loader=yaml.Loader)

    if args.dataset not in configs['LANE_DATASETS'].keys():
        raise ValueError

    mean = configs['GENERAL']['MEAN']
    std = configs['GENERAL']['STD']

    if args.task == 'lane':
        num_classes = configs[configs['LANE_DATASETS'][args.dataset]]['NUM_CLASSES']
        base = configs[configs['LANE_DATASETS'][args.dataset]]['BASE_DIR']
        if torch.cuda.is_available():
            device = torch.device('cuda:0')
        net = build_lane_model(args, num_classes)
        print(device)
        net.to(device)
        load_checkpoint(net=net, optimizer=None, lr_scheduler=None, filename=args.continue_from)
        input_size = (args.height, args.width)
        val_loader = lane_init(dataset=args.dataset, input_sizes=(args.height, args.width), mean=mean, std=std, base=base)
        fps = lane_speed_evaluate(net, device, val_loader, args.mixed_precision, input_size)
        print("fps:"+str(fps))
    elif args.task == 'segmentation':
        print("In progress...")
