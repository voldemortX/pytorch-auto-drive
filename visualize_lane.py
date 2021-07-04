# Redundant codes are kept for easier customized extensions/modifications
import yaml
import argparse
import torch
import cv2
from torch.cuda.amp import autocast
from tqdm import tqdm
from cv2 import VideoWriter_fourcc
from mmcv import VideoReader
from PIL import Image
from utils.datasets import CULane
from utils.all_utils_semseg import load_checkpoint
from utils.all_utils_landec import build_lane_detection_model
from utils.datasets import ImageFolderDataset
from tools.vis_tools import lane_detection_visualize_batched, simple_lane_detection_transform, \
    check_file_type, FileType, save_images, lane_inference
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
    parser.add_argument('--batch-size', type=int, default=1,
                        help='Batch size for inference with video/image folder (default: 1)')
    parser.add_argument('--workers', type=int, default=0,
                        help='Number of workers (default: 0)')
    args = parser.parse_args()
    with open('configs.yaml', 'r') as f:  # Safer and cleaner than box/EasyDict
        configs = yaml.load(f, Loader=yaml.Loader)

    if args.dataset not in configs['LANE_DATASETS'].keys():
        raise ValueError
    num_classes = configs[configs['LANE_DATASETS'][args.dataset]]['NUM_CLASSES']
    mask_colors = configs[configs['LANE_DATASETS'][args.dataset]]['COLORS']
    mask_colors = torch.tensor(mask_colors)
    keypoint_color = configs['VIS']['LANE_KEYPOINT_COLOR']
    image_suffix = configs['VIS']['IMAGE_SUFFIX']
    video_suffix = configs['VIS']['VIDEO_SUFFIX']
    image_type = check_file_type(args.image_path, image_suffix, video_suffix)

    if args.mask_path is not None or args.keypoint_path is not None:  # No inference
        images = Image.open(args.image_path).convert('RGB')
        images = F.to_tensor(images).unsqueeze(0)
        if args.mask_path is not None:
            masks = Image.open(args.mask_path)
            masks = ToTensor.label_to_tensor(masks).unsqueeze(0)
            assert masks.shape[-1] == images.shape[-1] and masks.shape[-2] == images.shape[-2], \
                'Mask and image shape mismatch!'
        else:
            masks = None
        if args.keypoint_path is not None:
            with open(args.keypoint_path, 'r') as f:
                keypoints = [CULane.load_target_xy(f.readlines())]
        else:
            keypoints = None

        results = lane_detection_visualize_batched(images, masks=masks, keypoints=keypoints,
                                                   mask_colors=mask_colors, keypoint_color=keypoint_color,
                                                   std=None, mean=None)
        save_images(results, [args.save_path])
    else:  # Inference needed
        # Prepare for inference
        mean = configs['GENERAL']['MEAN'] if args.mean is None else args.mean
        std = configs['GENERAL']['STD'] if args.std is None else args.std
        num_classes = configs[configs['LANE_DATASETS'][args.dataset]]['NUM_CLASSES']
        net = build_lane_detection_model(args, num_classes)
        images_trans = simple_lane_detection_transform(mean=mean, std=std, resize_shape=[args.height, args.width],
                                                       to_tensor=False if image_type == FileType.VIDEO else True)
        device = torch.device('cpu')
        if torch.cuda.is_available():
            device = torch.device('cuda:0')
        net = net.to(device)
        net.eval()
        mask_colors = mask_colors.to(device)
        mean = torch.tensor(mean, device=device)
        std = torch.tensor(std, device=device)
        load_checkpoint(net=net, optimizer=None, lr_scheduler=None, filename=args.continue_from)

        # Inference
        if image_type == FileType.DIR:  # Image folder (depth 1)
            if check_file_type(args.save_path, image_suffix, video_suffix) != FileType.DIR:
                raise ValueError('Must use a folder to save folder inference results!')
            dataset = ImageFolderDataset(root=args.image_path, output_dir=args.save_path, transforms=images_trans)
            loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=args.batch_size,
                                                 num_workers=args.workers, shuffle=False)
            with torch.no_grad():
                for images, original_images, filenames in tqdm(loader):
                    images = images.to(device)
                    original_images = original_images.to(device)
                    original_size = (original_images.shape[-2], original_images.shape[-1])
                    inference_size = (images.shape[-2], images.shape[-1])
                    keypoints = lane_inference(net, images, inference_size, original_size, args, configs)
                    results = lane_detection_visualize_batched(original_images, masks=None, keypoints=keypoints,
                                                               mask_colors=mask_colors, keypoint_color=keypoint_color,
                                                               std=None, mean=None)
                    save_images(results, filenames=filenames)
        elif image_type == FileType.IMAGE:  # Single image
            images = Image.open(args.image_path).convert('RGB')
            original_images = F.to_tensor(images).clone().unsqueeze(0)
            original_size = (original_images.shape[-2], original_images.shape[-1])
            images = images_trans(images).unsqueeze(0)
            inference_size = (images.shape[-2], images.shape[-1])
            images = images.to(device)
            original_images = original_images.to(device)
            keypoints = lane_inference(net, images, inference_size, original_size, args, configs)
            results = lane_detection_visualize_batched(original_images, masks=None, keypoints=keypoints,
                                                       mask_colors=mask_colors, keypoint_color=keypoint_color,
                                                       std=None, mean=None)
            save_images(results, filenames=[args.save_path])
        elif image_type == FileType.VIDEO:  # Single video
            video = VideoReader(args.image_path)
            writer = cv2.VideoWriter(args.save_path, VideoWriter_fourcc(*'XVID'), video.fps, video.resolution)
            original_size = (video.resolution[1], video.resolution[0])
            print('Total frames: {:d}'.format(len(video)))
            with torch.no_grad():
                for i in tqdm(range(len(video) // args.batch_size)):
                    images_numpy = video[i * args.batch_size: (i + 1) * args.batch_size]  # Numpy can suffer a index OOB
                    images = torch.stack([torch.from_numpy(img) for img in images_numpy])
                    images = images.permute(0, 3, 1, 2) / 255.0  # BHWC-rgb uint8 -> BCHW-rgb float
                    original_images = images.clone()
                    images = images_trans(images)
                    inference_size = (images.shape[-2], images.shape[-1])
                    images = images.to(device)
                    original_images = original_images.to(device)
                    keypoints = lane_inference(net, images, inference_size, original_size, args, configs)
                    results = lane_detection_visualize_batched(original_images, masks=None, keypoints=keypoints,
                                                               mask_colors=mask_colors, keypoint_color=keypoint_color,
                                                               std=None, mean=None)
                    for j in range(results.shape[0]):
                        writer.write(results[j])
            writer.release()
        else:
            raise ValueError('File must be an image, a video, or a directory!')
