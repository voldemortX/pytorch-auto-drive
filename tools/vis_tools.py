import os
import filetype
import numpy as np
import cv2
import torch
from enum import Enum
from PIL import Image
from transforms import ToTensor, Resize, ZeroPad, Normalize, Compose
from transforms import functional as F


# File mode statics
class FileType(Enum):
    DIR = 1
    IMAGE = 2
    VIDEO = 3


def tensor_image_to_numpy(images):

    return (images * 255.0).cpu().numpy().astype(np.uint8)


def save_images(images, filenames):
    # Save tensor images in range [0.0, 1.0]
    # filenames: List[str]
    assert images.shape[0] == len(filenames)
    np_results = tensor_image_to_numpy(images)
    for i in range(len(filenames)):
        Image.fromarray(np_results[i]).save(filenames[i])


def segmentation_visualize_batched(images, labels, colors, std=None, mean=None, trans=0.3, ignore_color=None):
    # Draw images + labels from tensors (batched)
    # images (4D), labels (3D), colors (2D), std, mean, ignore_color: torch.Tensor
    # trans: how transparent is the label
    # ignore_color: in range [0.0, 1.0]
    assert images.shape[0] == labels.shape[0]

    # Map label to RGB (N, d1, d2) = {0~20, 255} => (N, d1, d2, 3) = {0.0~1.0}
    if colors is None:  # Same color (white) for all classes
        colors = torch.tensor([[0, 0, 0], [255, 255, 255]], device=images.device)
        labels[labels > 0] = 1
    else:
        labels[labels == 255] = colors.shape[0] - 1  # Color for ignore
    labels = colors[labels] / 255.0

    # Denormalize if needed and map from (N, 3, d1, d2) to (N, d1, d2, 3)
    images = images.permute(0, 2, 3, 1)
    if std is not None and mean is not None:
        images = (images.float() * std + mean).clamp_(0.0, 1.0)

    # Mix (should not need another clamp)
    results = images * trans + labels * (1 - trans)
    if ignore_color is not None:
        filter_mask = (labels == ignore_color).sum(dim=-1, keepdim=True) == ignore_color.shape[0]
        results = results * ~filter_mask + images * filter_mask

    return results


def lane_detection_visualize_batched(images, filenames, masks=None, keypoints=None,
                                     mask_colors=None, keypoint_color=None, std=None, mean=None):
    # Draw images + lanes from tensors (batched)
    # None masks/keypoints and keypoints (x < 0 or y < 0) will be ignored
    # images (4D), masks (3D), keypoints (4D), colors (2D), std, mean: torch.Tensor
    # keypoints can be either List[List[N x 2 numpy array]] (for variate length lanes) or a 4D numpy array
    # filenames: List[str]
    # keypoint_color: BGR
    if masks is not None:
        images = segmentation_visualize_batched(images, masks, mask_colors, std, mean,
                                                trans=0, ignore_color=mask_colors[0])
    if keypoints is not None:
        # TODO: Get rid of cv2
        if masks is None:
            images = images.permute(0, 2, 3, 1)
        if std is not None and mean is not None:
            images = (images.float() * std + mean)
        images = images.clamp_(0.0, 1.0) * 255.0
        images = images[..., [2, 1, 0]].cpu().numpy().astype(np.uint8)
        if keypoint_color is None:
            keypoint_color = [0, 0, 255]  # BGR: Red (sits well with lane colors)
        for i in range(len(filenames)):
            for j in range(len(keypoints[i])):
                temp = keypoints[i][j][(keypoints[i][j][:, 0] > 0) * (keypoints[i][j][:, 1] > 0)]
                # Draw solid keypoints
                for k in range(temp.shape[0]):
                    cv2.circle(images[i], (int(temp[k][0]), int(temp[k][1])),
                               radius=3, color=keypoint_color, thickness=-1)
            cv2.imwrite(filenames[i], images[i])
    else:
        save_images(images=images, filenames=filenames)


def simple_segmentation_transform(resize_shape, mean, std, dataset='voc', city_aug=0, to_tensor=True):
    # city_aug correspond to city_aug in init()
    # Assume images in B x C x H x W
    # resize_shape: list[int]
    transforms = [ToTensor()] if to_tensor else []
    if dataset == 'voc':
        transforms.append(ZeroPad(size=resize_shape))
    else:
        transforms.append(Resize(size_image=resize_shape, size_label=resize_shape))

    if city_aug != 2:  # No normalization for ERFNet
        transforms.append(Normalize(mean=mean, std=std))

    return Compose(transforms)


# Segmentation methods have simple and unified output formats,
# same simple post-process will suffice
def unified_segmentation_label_formatting(labels, original_size, args):
    if args.dataset == 'voc':
        labels = torch.nn.functional.interpolate(labels, size=(args.height, args.width),
                                                 mode='bilinear', align_corners=True)
        labels = F.crop(labels, 0, 0, original_size[0], original_size[1])
    else:
        labels = torch.nn.functional.interpolate(labels, size=original_size, mode='bilinear',
                                                 align_corners=True)
    return labels.argmax(1)


def simple_lane_detection_transform(images, resize_shape, mean, std):
    # Assume images in B x C x H x W
    # resize_shape: list[int]
    pass


# Return file type (directory-1/image-2/video-3) based on suffix
def check_file_type(filename, image_suffixes, video_suffixes):
    if os.path.isdir(filename):
        return FileType.DIR
    else:
        filetype_str = filetype.guess(filename).extension
        if filetype_str in image_suffixes:
            return FileType.IMAGE
        elif filetype_str in video_suffixes:
            return FileType.VIDEO
        else:
            raise ValueError('File must be an image, a video, or a directory!')
