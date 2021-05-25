import numpy as np


def crop(points, top, left, height, width, ignore_x=-2):
    # Crop a np.array (L x N x 2) of points (x, y), original axis start from top-left corner
    # Essentially a translation with filtering, consider only crop area within the image
    ignore_filter = (points[:, :, 0] == ignore_x)
    points -= np.array([left, top], dtype=points.dtype)  # translation
    ignore_by_crop = (points[:, :, 0] < 0) + (points[:, :, 0] > width) \
                     + (points[:, :, 1] < 0) + (points[:, :, 1] > height)  # filtering
    ignore_filter += ignore_by_crop
    points[:, :, 0] = points[:, :, 0] * ~ignore_filter + (-2) * ignore_filter

    return points
