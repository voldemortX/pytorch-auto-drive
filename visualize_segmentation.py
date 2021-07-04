# TODO: DOC, pip & dataset structure
import yaml
import argparse
import torch
import cv2
from cv2 import VideoWriter_fourcc
from mmcv import VideoReader
from torch.cuda.amp import autocast
from PIL import Image
from tqdm import tqdm
from utils.all_utils_semseg import load_checkpoint, build_segmentation_model
from utils.datasets import ImageFolderDataset
from tools.vis_tools import segmentation_visualize_batched, simple_segmentation_transform, save_images, \
    check_file_type, FileType, unified_segmentation_label_formatting, tensor_image_to_numpy
from transforms import functional as F
from transforms.transforms import ToTensor


if __name__ == '__main__':
    # Settings
    parser = argparse.ArgumentParser(description='PyTorch Auto-drive')
    parser.add_argument('--image-path', type=str, default='test_image.png',
                        help='Image/Video/Image directory input path (default: test_image.png)')
    parser.add_argument('--mask-path', type=str, default=None,
                        help='Mask input path, if set to None, inference will be performed (default: None)')
    parser.add_argument('--save-path', type=str, default='test_result.png',
                        help='Result output path (default: test_result.png)')
    parser.add_argument('--height', type=int, default=505,
                        help='Image input height (default: 505)')
    parser.add_argument('--width', type=int, default=505,
                        help='Image input width (default: 505)')
    parser.add_argument('--mean', metavar='N', type=float, nargs='+',
                        help='RGB mean for input data (default: ImageNet mean)')
    parser.add_argument('--std', metavar='N', type=float, nargs='+',
                        help='RGB standard variance for input data (default: ImageNet std)')
    parser.add_argument('--model', type=str, default='deeplabv3',
                        help='Model selection (fcn/erfnet/deeplabv2/deeplabv3/enet) (default: deeplabv3)')
    parser.add_argument('--dataset', type=str, default='voc',
                        help='Color statistics from which dataset,'
                             'PASCAL VOC 2012(voc)/Cityscapes(city)/GTAV(gtav)/SYNTHIA(synthia) (default: voc)')
    parser.add_argument('--map-id', action='store_true', default=False,
                        help='Enable label_id_map, usually for ground truth mask in Cityscapes (default: False)')
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

    if args.dataset not in configs['SEGMENTATION_DATASETS'].keys():
        raise ValueError('Invalid dataset name! Try use lower cases, and city for cityscapes.')
    colors = configs[configs['SEGMENTATION_DATASETS'][args.dataset]]['COLORS']
    colors = torch.tensor(colors)
    image_suffix = configs['VIS']['IMAGE_SUFFIX']
    video_suffix = configs['VIS']['VIDEO_SUFFIX']
    image_type = check_file_type(args.image_path, image_suffix, video_suffix)

    if args.mask_path is not None:  # User provided mask (no need for inference)
        if image_type != FileType.IMAGE:
            raise ValueError('You can only provide mask for single image!')
        images = Image.open(args.image_path).convert('RGB')
        images = F.to_tensor(images).unsqueeze(0)
        labels = Image.open(args.mask_path)
        labels = ToTensor.label_to_tensor(labels).unsqueeze(0)
        assert labels.shape[-1] == images.shape[-1] and labels.shape[-2] == images.shape[-2], \
            'Mask and image shape mismatch!'
        if args.map_id:
            label_id_map = torch.tensor(configs[configs['SEGMENTATION_DATASETS'][args.dataset]]['LABEL_ID_MAP'])
            labels[labels >= label_id_map.shape[0]] = 0
            labels = label_id_map[labels]
        results = segmentation_visualize_batched(images=images, labels=labels, colors=colors, mean=None, std=None)
        save_images(results, filenames=[args.save_path])
    else:  # Inference needed
        # Prepare for inference
        mean = configs['GENERAL']['MEAN'] if args.mean is None else args.mean
        std = configs['GENERAL']['STD'] if args.std is None else args.std
        input_sizes = configs[configs['SEGMENTATION_DATASETS'][args.dataset]]['SIZES']
        num_classes = configs[configs['SEGMENTATION_DATASETS'][args.dataset]]['NUM_CLASSES']
        net, city_aug, _, weights = build_segmentation_model(configs, args, num_classes, 0, input_sizes)
        images_trans = simple_segmentation_transform(mean=mean, std=std, resize_shape=[args.height, args.width],
                                                     dataset=args.dataset, city_aug=city_aug,
                                                     to_tensor=False if image_type == FileType.VIDEO else True)
        device = torch.device('cpu')
        if torch.cuda.is_available():
            device = torch.device('cuda:0')
        if weights is not None:
            weights = weights.to(device)
        net = net.to(device)
        net.eval()
        colors = colors.to(device)
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
                    with autocast(args.mixed_precision):
                        labels = net(images)['out']
                    original_size = (original_images.shape[-2], original_images.shape[-1])
                    labels = unified_segmentation_label_formatting(labels, original_size=original_size, args=args)
                    results = segmentation_visualize_batched(images=original_images, labels=labels,
                                                             colors=colors, mean=None, std=None)
                    save_images(results, filenames=filenames)
        elif image_type == FileType.IMAGE:  # Single image
            images = Image.open(args.image_path).convert('RGB')
            original_images = F.to_tensor(images).clone().unsqueeze(0)
            original_size = (original_images.shape[-2], original_images.shape[-1])
            images = images_trans(images).unsqueeze(0)
            images = images.to(device)
            original_images = original_images.to(device)
            with autocast(args.mixed_precision):
                labels = net(images)['out']
            labels = unified_segmentation_label_formatting(labels, original_size, args)
            results = segmentation_visualize_batched(images=original_images, labels=labels,
                                                     colors=colors, mean=None, std=None)
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
                    images = images.to(device)
                    original_images = original_images.to(device)
                    with autocast(args.mixed_precision):
                        labels = net(images)['out']
                    labels = unified_segmentation_label_formatting(labels, original_size=original_size, args=args)
                    results = segmentation_visualize_batched(images=original_images, labels=labels,
                                                             colors=colors, mean=None, std=None)
                    np_results = tensor_image_to_numpy(results)
                    for j in range(np_results.shape[0]):
                        writer.write(np_results[j])
            writer.release()
        else:
            raise ValueError('File must be an image, a video, or a directory!')
