import numpy as np
import transforms.functional as F
from PIL import Image


def segmentation_visualize_batched(images, labels, filenames, colors, num_classes, std=None, mean=None, trans=0.3):
    # Draw images + labels from tensors (batched)
    # images (4D), labels (3D), colors (2D), std, mean: torch.Tensor
    # filenames: List[str]
    # trans: How transparent is the label
    assert images.shape[0] == labels.shape[0] == len(filenames)

    # Map label to RGB (N, d1, d2) = {0~20, 255} => (N, d1, d2, 3) = {0.0~1.0}
    labels[labels == 255] = num_classes  # Color for ignore
    labels = colors[labels] / 255.0

    # Denormalize if needed and map from (N, 3, d1, d2) to (N, d1, d2, 3)
    images = images.permute(0, 2, 3, 1)
    if std is not None and mean is not None:
        images = (images.float() * std + mean).clamp_(0.0, 1.0)

    # Mix (should not need another clamp)
    results = images * trans + labels * (1 - trans)

    # Save
    np_results = (results * 255.0).cpu().numpy().astype(np.uint8)
    for i in range(len(filenames)):
        Image.fromarray(np_results[i]).save(filenames[i])


def simple_segmentation_transform(images, resize_shape, mean, std, dataset='voc', city_aug=0):
    # city_aug correspond to city_aug in init()
    # Assume images in B x C x H x W
    # resize_shape: list[int]
    if dataset == 'voc':
        images = F.pad(images, [0, 0, resize_shape[1] - images.shape[-1], resize_shape[0] - images.shape[-2]], fill=0)
    else:
        images = F.resize(images, resize_shape, interpolation=Image.LINEAR)

    if city_aug != 2:  # No normalization for ERFNet
        images = F.normalize(images, mean=mean, std=std)

    return images
