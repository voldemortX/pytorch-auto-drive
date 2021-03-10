import yaml
import argparse
import torch
from torch.cuda.amp import autocast
from PIL import Image
from utils.datasets import CULane
from utils.all_utils_semseg import load_checkpoint
from tools.vis_tools import lane_detection_visualize_batched, simple_lane_detection_transform
from transforms import functional as F
from transforms.transforms import ToTensor


if __name__ == '__main__':
    # Settings
    parser = argparse.ArgumentParser(description='PyTorch Auto-drive')
    parser.add_argument('--image-path', type=str, default='test_image.png',
                        help='Image input path (default: test_image.png)')
    parser.add_argument('--mask-path', type=str, default=None,
                        help='Mask input path, if both mask & keypoint are None,'
                             'inference will be performed (default: None)')
    parser.add_argument('--keypoint-path', type=str, default=None,
                        help='Keypoint input path (expect json file in CULane format, [x, y]),'
                             'if both mask & keypoint are None, inference will be performed (default: None)')
    parser.add_argument('--save-path', type=str, default='test_result.png',
                        help='Result output path (default: test_result.png)')
    parser.add_argument('--height', type=int, default=288,
                        help='Image input height (default: 288)')
    parser.add_argument('--width', type=int, default=800,
                        help='Image input width (default: 800)')
    parser.add_argument('--mean', metavar='N', type=float, nargs='+',
                        help='RGB mean for input data (default: ImageNet mean)')
    parser.add_argument('--std', metavar='N', type=float, nargs='+',
                        help='RGB standard variance for input data (default: ImageNet std)')
    parser.add_argument('--dataset', type=str, default='tusimple',
                        help='Train/Evaluate on TuSimple (tusimple) / CULane (culane) (default: tusimple)')
    parser.add_argument('--method', type=str, default='baseline',
                        help='method selection (lstr/scnn/sad/baseline/etc) (default: baseline)')
    parser.add_argument('--backbone', type=str, default='erfnet',
                        help='backbone selection (erfnet/vgg16/resnet18s/resnet18/resnet34/resnet50/resnet101)'
                             '(default: erfnet)')
    parser.add_argument('--mixed-precision', action='store_true', default=False,
                        help='Enable mixed precision training (default: False)')
    parser.add_argument('--continue-from', type=str, default=None,
                        help='Continue training from a previous checkpoint')
    args = parser.parse_args()
    with open('configs.yaml', 'r') as f:  # Safer and cleaner than box/EasyDict
        configs = yaml.load(f, Loader=yaml.Loader)

    if args.dataset not in configs['LANE_DATASETS'].keys():
        raise ValueError
    num_classes = configs[configs['LANE_DATASETS'][args.dataset]]['NUM_CLASSES']
    mask_colors = configs[configs['LANE_DATASETS'][args.dataset]]['COLORS']
    mask_colors = torch.tensor(mask_colors)
    images = Image.open(args.image_path).convert('RGB')
    images = F.to_tensor(images).unsqueeze(0)

    if args.mask_path is None and args.keypoint_path is None:  # Inference
        print('Inference is not supported yet.')
        raise NotImplementedError
    else:  # No inference
        if args.mask_path is not None:
            masks = Image.open(args.mask_path)
            masks = ToTensor.label_to_tensor(masks).unsqueeze(0)
            assert images.shape[-1] == masks.shape[-1]
            assert images.shape[-2] == masks.shape[-2]
        else:
            masks = None
        if args.keypoint_path is not None:
            with open(args.keypoint_path, 'r') as f:
                keypoints = [CULane.load_target_xy(f.readlines())]
        else:
            keypoints = None

    lane_detection_visualize_batched(images, [args.save_path], masks=masks, keypoints=keypoints,
                                     mask_colors=mask_colors, keypoint_color=None, std=None, mean=None)
