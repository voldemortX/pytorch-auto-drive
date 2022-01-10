import math
import torch
import numpy as np


def crop(points, top, left, height, width, ignore_x=-2):
    # Crop a np.array (L x N x 2) of points (x, y), original axis start from top-left corner
    # Essentially a translation with filtering, consider only crop area within the image
    # Set ignore_x to None if you don't want to ignore out-of-image points
    if sum(points.shape) == 0:
        return points
    if ignore_x is not None:
        ignore_filter = (points[:, :, 0] == ignore_x)

    points -= np.array([left, top], dtype=points.dtype)  # translation

    if ignore_x is not None:
        ignore_by_crop = (points[:, :, 0] < 0) + (points[:, :, 0] > width) \
            + (points[:, :, 1] < 0) + (points[:, :, 1] > height)  # filtering
        ignore_filter += ignore_by_crop
        points[:, :, 0] = points[:, :, 0] * ~ignore_filter + ignore_x * ignore_filter

    return points


def resize(points, in_size, out_size, ignore_x=-2):
    # Resize a np.array (L x N x 2) of points (x, y), original axis start from top-left corner
    # x <-> w, y <-> h
    # Set ignore_x to None if you don't want to ignore out-of-image points
    if sum(points.shape) == 0:
        return points
    if ignore_x is not None:
        ignore_filter = (points[:, :, 0] == ignore_x)

    in_h, in_w = in_size
    out_h, out_w = out_size
    scale = np.array([out_w / in_w, out_h / in_h], dtype=np.float32)
    points = points * scale

    if ignore_x is not None:
        points[:, :, 0] = points[:, :, 0] * ~ignore_filter + ignore_x * ignore_filter

    return points


def hflip(points, mid_x, ignore_x=-2):
    # Flip a np.array (L x N x 2) of points (x, y) horizontally, original axis start from top-left corner
    # Set ignore_x to None if you don't want to ignore out-of-image points
    if sum(points.shape) == 0:
        return points
    if ignore_x is not None:
        ignore_filter = (points[:, :, 0] == ignore_x)

    points[:, :, 0] = 2 * mid_x - points[:, :, 0]

    if ignore_x is not None:
        points[:, :, 0] = points[:, :, 0] * ~ignore_filter + ignore_x * ignore_filter

    return points


def rotate(points, angle, h, w, ignore_x=-2):
    # Rotate a np.array (L x N x 2) of points (x, y) anti-clockwise, original axis start from top-left corner
    # Set ignore_x to None if you don't want to ignore out-of-image points
    if sum(points.shape) == 0:
        return points
    if ignore_x is not None:
        ignore_filter = (points[:, :, 0] == ignore_x)

    offset = np.array([w / 2, h / 2], dtype=np.float32)
    matrix = np.array([[math.cos(angle / 180.0 * math.pi), math.sin(-angle / 180.0 * math.pi)],
                       [math.sin(angle / 180.0 * math.pi), math.cos(angle / 180.0 * math.pi)]], dtype=np.float32)
    points = np.matmul((points - offset), matrix) + offset

    if ignore_x is not None:
        # exceed border
        ignore_filter += ((points[:, :, 0] > w) + (points[:, :, 1] > h) + ((points > 0).sum(axis=-1) < 2))
        points[:, :, 0] = points[:, :, 0] * ~ignore_filter + ignore_x * ignore_filter

    return points


def normalize(points, h, w, ignore_x=-2):
    # Divide keypoints by h & w to 0~1
    # Note that normalize works with tensors (L x N x 2)
    # A special case of resize
    # Set ignore_x to None if you don't want to ignore out-of-image points
    if sum(points.shape) == 0:
        return points
    points = points / torch.tensor([w, h], device=points.device, dtype=points.dtype)

    if ignore_x is not None:
        points[points[:, :, 0] < 0][:, 0] = ignore_x

    return points
