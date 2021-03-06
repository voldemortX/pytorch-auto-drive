import yaml
import argparse
import torch
from torch.cuda.amp import autocast
from PIL import Image
from utils.all_utils_semseg import load_checkpoint, build_segmentation_model
from tools.vis_tools import segmentation_visualize_batched, simple_segmentation_transform, save_images
from transforms import functional as F
from transforms.transforms import ToTensor


if __name__ == '__main__':
    # Settings
    parser = argparse.ArgumentParser(description='PyTorch Auto-drive')
    parser.add_argument('--image-path', type=str, default='test_image.png',
                        help='Image input path (default: test_image.png)')
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
    args = parser.parse_args()
    with open('configs.yaml', 'r') as f:  # Safer and cleaner than box/EasyDict
        configs = yaml.load(f, Loader=yaml.Loader)

    if args.dataset not in configs['SEGMENTATION_DATASETS'].keys():
        raise ValueError
    colors = configs[configs['SEGMENTATION_DATASETS'][args.dataset]]['COLORS']
    colors = torch.tensor(colors)
    images = Image.open(args.image_path).convert('RGB')
    images = F.to_tensor(images).unsqueeze(0)

    if args.mask_path is None:  # Inference
        original_size = (images.shape[-2], images.shape[-1])
        mean = configs['GENERAL']['MEAN'] if args.mean is None else args.mean
        std = configs['GENERAL']['STD'] if args.std is None else args.std
        input_sizes = configs[configs['SEGMENTATION_DATASETS'][args.dataset]]['SIZES']
        num_classes = configs[configs['SEGMENTATION_DATASETS'][args.dataset]]['NUM_CLASSES']
        assert len(mean) == len(std)
        net, city_aug, _, weights = build_segmentation_model(configs, args, num_classes, 0, input_sizes)
        images_trans = simple_segmentation_transform(images, mean=mean, std=std, resize_shape=[args.height, args.width],
                                                     dataset=args.dataset, city_aug=city_aug)
        device = torch.device('cpu')
        if torch.cuda.is_available():
            device = torch.device('cuda:0')
        if weights is not None:
            weights = weights.to(device)
        images_trans = images_trans.to(device)
        images = images.to(device)
        net = net.to(device)
        colors = colors.to(device)
        mean = torch.tensor(mean, device=device)
        std = torch.tensor(std, device=device)
        load_checkpoint(net=net, optimizer=None, lr_scheduler=None, filename=args.continue_from)
        with autocast(args.mixed_precision):
            labels = net(images_trans)['out']
        if args.dataset == 'voc':
            labels = torch.nn.functional.interpolate(labels, size=(args.height, args.width),
                                                     mode='bilinear', align_corners=True)
            labels = F.crop(labels, 0, 0, original_size[0], original_size[1])
        else:
            labels = torch.nn.functional.interpolate(labels, size=original_size, mode='bilinear', align_corners=True)
        labels = labels.argmax(1)
    else:  # No inference
        labels = Image.open(args.mask_path)
        labels = ToTensor.label_to_tensor(labels).unsqueeze(0)
        assert labels.shape[-1] == images.shape[-1]
        assert labels.shape[-2] == images.shape[-2]
        if args.map_id:
            label_id_map = torch.tensor(configs[configs['SEGMENTATION_DATASETS'][args.dataset]]['LABEL_ID_MAP'])
            labels[labels >= label_id_map.shape[0]] = 0
            labels = label_id_map[labels]

    results = segmentation_visualize_batched(images=images, labels=labels, colors=colors, mean=None, std=None)
    save_images(results, filenames=[args.save_path])
