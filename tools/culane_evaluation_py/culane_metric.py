# Copied from Turoad/lanedet
# Slightly differs from official metric, recommend using this only for visualization
import cv2
import numpy as np
from scipy.interpolate import splprep, splev
from scipy.optimize import linear_sum_assignment
from shapely.geometry import LineString, Polygon


def draw_lane(lane, img=None, img_shape=None, width=30):
    if img is None:
        img = np.zeros(img_shape, dtype=np.uint8)
    lane = lane.astype(np.int32)
    for p1, p2 in zip(lane[:-1], lane[1:]):
        cv2.line(img, tuple(p1), tuple(p2), color=1, thickness=width)
    return img


def discrete_cross_iou(xs, ys, width=30, img_shape=(590, 1640, 3)):
    xs = [draw_lane(lane, img_shape=img_shape[:2], width=width) for lane in xs]
    ys = [draw_lane(lane, img_shape=img_shape[:2], width=width) for lane in ys]

    ious = np.zeros((len(xs), len(ys)))
    for i, x in enumerate(xs):
        for j, y in enumerate(ys):
            inter = (x * y).sum()
            ious[i, j] = inter / (x.sum() + y.sum() - inter)
    return ious


def continuous_cross_iou(xs, ys, width=30, img_shape=(590, 1640, 3)):
    h, w, _ = img_shape
    image = Polygon([(0, 0), (0, h - 1), (w - 1, h - 1), (w - 1, 0)])
    xs = [LineString(lane).buffer(distance=width / 2., cap_style=1, join_style=2).intersection(image) for lane in xs]
    ys = [LineString(lane).buffer(distance=width / 2., cap_style=1, join_style=2).intersection(image) for lane in ys]

    ious = np.zeros((len(xs), len(ys)))
    for i, x in enumerate(xs):
        for j, y in enumerate(ys):
            ious[i, j] = x.intersection(y).area / x.union(y).area

    return ious


def interp(points, n=50):
    if len(points) == 2:
        return np.array(points)
    x = [x for x, _ in points]
    y = [y for _, y in points]
    tck, u = splprep([x, y], s=0, t=n, k=min(3, len(points) - 1))

    u = np.linspace(0., 1., num=(len(u) - 1) * n + 1)
    return np.array(splev(u, tck)).T


def culane_metric(pred, anno, width=30, iou_threshold=0.5, official=True, img_shape=(590, 1640, 3)):
    if len(pred) == 0:
        return 0, 0, len(anno), np.zeros(len(pred)), np.zeros(len(pred), dtype=bool)
    if len(anno) == 0:
        return 0, len(pred), 0, np.zeros(len(pred)), np.zeros(len(pred), dtype=bool)
    interp_pred = [interp(pred_lane) for pred_lane in pred]  # (4, 50, 2)
    interp_anno = [interp(anno_lane) for anno_lane in anno]  # (4, 50, 2)

    if official:
        ious = discrete_cross_iou(interp_pred, interp_anno, width=width, img_shape=img_shape)
    else:
        ious = continuous_cross_iou(interp_pred, interp_anno, width=width, img_shape=img_shape)

    row_ind, col_ind = linear_sum_assignment(1 - ious)
    tp = int((ious[row_ind, col_ind] > iou_threshold).sum())
    fp = len(pred) - tp
    fn = len(anno) - tp
    pred_ious = np.zeros(len(pred))
    pred_ious[row_ind] = ious[row_ind, col_ind]
    return tp, fp, fn, pred_ious, pred_ious > iou_threshold
