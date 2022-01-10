import cv2
import torch
import numpy as np
try:
    import ujson as json
except ImportError:
    import json

from .models.lane_detection.utils import lane_pruning


# Adapted from harryhan618/SCNN_Pytorch
# Note that in tensors we have indices start from 0 and in annotations coordinates start at 1
def get_lane(prob_map, gap, ppl, thresh, resize_shape=None, dataset='culane'):
    """
    Arguments:
    ----------
    prob_map: prob map for single lane, np array size (h, w)
    resize_shape:  reshape size target, (H, W)
    Return:
    ----------
    coords: x coords bottom up every gap px, 0 for non-exist, in resized shape
    """

    if resize_shape is None:
        resize_shape = prob_map.shape
    h, w = prob_map.shape
    H, W = resize_shape
    coords = np.zeros(ppl)
    for i in range(ppl):
        if dataset == 'tusimple':  # Annotation start at 10 pixel away from bottom
            y = int(h - (ppl - i) * gap / H * h)
        elif dataset in ['culane', 'llamas']:  # Annotation start at bottom
            y = int(h - i * gap / H * h - 1)  # Same as original SCNN code
        else:
            raise ValueError
        if y < 0:
            break
        line = prob_map[y, :]
        id = np.argmax(line)
        if line[id] > thresh:
            coords[i] = int(id / w * W)
    if (coords > 0).sum() < 2:
        coords = np.zeros(ppl)
    return coords


# Adapted from harryhan618/SCNN_Pytorch
def prob_to_lines(seg_pred, exist, resize_shape=None, smooth=True, gap=20, ppl=None, thresh=0.3, dataset='culane'):
    """
    Arguments:
    ----------
    seg_pred: np.array size (num_classes, h, w)
    resize_shape:  reshape size target, (H, W)
    exist:   list of existence, e.g. [0, 1, 1, 0]
    smooth:  whether to smooth the probability or not
    gap: y pixel gap for sampling
    ppl:     how many points for one lane
    thresh:  probability threshold
    all_points: Whether to save all sample points or just points predicted as lane
    Return:
    ----------
    coordinates: [x, y] list of lanes, e.g.: [ [[9, 569], [50, 549]] ,[[630, 569], [647, 549]] ]
    """
    if resize_shape is None:
        resize_shape = seg_pred.shape[1:]  # seg_pred (num_classes, h, w)
    _, h, w = seg_pred.shape
    H, W = resize_shape
    coordinates = []

    if ppl is None:
        ppl = round(H / 2 / gap)

    for i in range(1, seg_pred.shape[0]):
        prob_map = seg_pred[i, :, :]
        if exist[i - 1]:
            if smooth:
                prob_map = cv2.blur(prob_map, (9, 9), borderType=cv2.BORDER_REPLICATE)
            coords = get_lane(prob_map, gap, ppl, thresh, resize_shape, dataset=dataset)
            if coords.sum() == 0:
                continue
            if dataset == 'tusimple':  # Invalid sample points need to be included as negative value, e.g. -2
                coordinates.append([[coords[j], H - (ppl - j) * gap] if coords[j] > 0 else [-2,  H - (ppl - j) * gap]
                                    for j in range(ppl)])
            elif dataset in ['culane', 'llamas']:
                coordinates.append([[coords[j], H - j * gap - 1] for j in range(ppl) if coords[j] > 0])
            else:
                raise ValueError

    return coordinates


# A unified inference function, for segmentation-based lane detection methods
@torch.no_grad()
def lane_as_segmentation_inference(net, inputs, input_sizes, gap, ppl, thresh, dataset, max_lane=0, forward=True):
    # Assume net and images are on the same device
    # images: B x C x H x W
    # Return: a list of lane predictions on each image
    outputs = net(inputs) if forward else inputs  # Support no forwarding inside this function
    prob_map = torch.nn.functional.interpolate(outputs['out'], size=input_sizes[0], mode='bilinear',
                                               align_corners=True).softmax(dim=1)
    existence_conf = outputs['lane'].sigmoid()
    existence = existence_conf > 0.5
    if max_lane != 0:  # Lane max number prior for testing
        existence, existence_conf = lane_pruning(existence, existence_conf, max_lane=max_lane)

    prob_map = prob_map.cpu().numpy()
    existence = existence.cpu().numpy()

    # Get coordinates for lanes
    lane_coordinates = []
    for j in range(existence.shape[0]):
        lane_coordinates.append(prob_to_lines(prob_map[j], existence[j], resize_shape=input_sizes[1],
                                              gap=gap, ppl=ppl, thresh=thresh, dataset=dataset))

    return lane_coordinates
