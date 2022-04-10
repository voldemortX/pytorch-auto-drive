import numpy as np
import cv2
import torch
from PIL import Image
from importmagician import import_from
with import_from('./'):
    from tools.culane_evaluation_py.culane_metric import culane_metric
    from tools.tusimple_evaluation.lane import LaneEval


# Colors statics
BGR_RED = [0, 0, 255]
BGR_GREEN = [0, 255, 0]
BGR_BLUE = [255, 0, 0]


def tensor_image_to_numpy(images):

    return (images * 255.0).cpu().numpy().astype(np.uint8)


def save_images(images, filenames):
    # Save tensor images in range [0.0, 1.0]
    # filenames: List[str]
    assert images.shape[0] == len(filenames)
    if type(images) != np.ndarray:  # Flexible
        images = tensor_image_to_numpy(images)
    for i in range(len(filenames)):
        Image.fromarray(images[i]).save(filenames[i])


def segmentation_visualize_batched(images, labels, colors, std=None, mean=None, trans=0.3, ignore_color=None,
                                   auto_color=True, ignore_index=255):
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
        ignore_pixels = labels == ignore_index
        bg_pixels = labels == 0
        if auto_color:  # Iterate colors (except background and ignore)
            labels = (labels - 1) % (colors.shape[0] - 2) + 1
        labels[ignore_pixels] = colors.shape[0] - 1  # Color for ignore
        labels[bg_pixels] = 0
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


def draw_points(image, points, colors, radius=5, thickness=-1):
    # Draw lines (defined by points) on an image as keypoints
    # colors: can be a list that defines different colors for each line
    for j in range(len(points)):
        temp = points[j][(points[j][:, 0] > 0) * (points[j][:, 1] > 0)]
        for k in range(temp.shape[0]):
            color = colors[j] if isinstance(colors[0], list) else colors
            cv2.circle(image, (int(temp[k][0]), int(temp[k][1])),
                       radius=radius, color=color, thickness=thickness)
    return image


def draw_points_as_lines(image, points, colors, thickness=3):
    # Draw lines (defined by points) on an image by connecting points to lines
    # colors: can be a list that defines different colors for each line
    for j in range(len(points)):
        temp = points[j][(points[j][:, 0] > 0) * (points[j][:, 1] > 0)]
        for k in range(temp.shape[0] - 1):
            color = colors[j] if isinstance(colors[0], list) else colors
            cv2.line(image,
                     (int(temp[k][0]), int(temp[k][1])),
                     (int(temp[k + 1][0]), int(temp[k + 1][1])),
                     color=color, thickness=thickness)
    return image


def lane_detection_visualize_batched(images, masks=None, keypoints=None,
                                     mask_colors=None, keypoint_color=None, std=None, mean=None,
                                     control_points=None, gt_keypoints=None, style='point', line_trans=0.4,
                                     compare_gt_metric='culane'):
    # Draw images + lanes from tensors (batched)
    # None masks/keypoints and keypoints (x < 0 or y < 0) will be ignored
    # images (4D), masks (3D), keypoints (4D), colors (2D), std, mean: torch.Tensor
    # keypoints can be either List[List[N x 2 numpy array]] (for variate length lanes) or a 4D numpy array
    # filenames: List[str]
    # keypoint_color: RGB
    if masks is not None:
        images = segmentation_visualize_batched(images, masks, mask_colors, std, mean,
                                                trans=0, ignore_color=mask_colors[0])
    if keypoints is not None:
        if masks is None:
            images = images.permute(0, 2, 3, 1)
        if std is not None and mean is not None:
            images = (images.float() * std + mean)
        images = images.clamp_(0.0, 1.0) * 255.0
        images = images[..., [2, 1, 0]].cpu().numpy().astype(np.uint8)
        if keypoint_color is None:
            keypoint_color = [0, 0, 0]  # Black (sits well with lane colors)
        else:
            keypoint_color = keypoint_color[::-1]  # To BGR

        # Draw
        for i in range(images.shape[0]):
            # Compare with GT
            if gt_keypoints is not None:
                if compare_gt_metric == 'culane':
                    tp, fp, fn, pred_ious, _ = culane_metric(keypoints[i], gt_keypoints[i])
                    keypoint_color = [BGR_GREEN if iou >= 0.5 else BGR_RED for iou in pred_ious]
                elif compare_gt_metric == 'tusimple':
                    x_pred = [keypoints[i][j][:, 0] for j in range(len(keypoints[i]))]
                    x_gt = [gt_keypoints[i][j][:, 0] for j in range(len(gt_keypoints[i]))]
                    y = gt_keypoints[i][0][:, 1].tolist()
                    acc, fp, fn, match, _ = LaneEval.bench_with_matches(x_pred, x_gt, y)
                    keypoint_color = [BGR_GREEN if m else BGR_RED for m in match]

            if style == 'point':
                if gt_keypoints is not None:
                    images[i] = draw_points(images[i], gt_keypoints[i], BGR_BLUE)
                images[i] = draw_points(images[i], keypoints[i], keypoint_color)
            elif style in ['line', 'bezier']:
                overlay = images[i].copy()
                if gt_keypoints is not None:
                    overlay = draw_points_as_lines(overlay, gt_keypoints[i], BGR_BLUE)
                overlay = draw_points_as_lines(overlay, keypoints[i], keypoint_color)
                images[i] = (images[i].astype(np.float) * line_trans +
                             overlay.astype(np.float) * (1 - line_trans)).astype(np.uint8)
                if style == 'bezier':
                    assert control_points is not None, 'Must provide control points for style bezier!'
                    images[i] = draw_points(images[i], control_points[i], keypoint_color)
            else:
                raise ValueError('Unknown keypoint visualization style: {}\nPlease use point/line/bezier'.format(style))
        images = images[..., [2, 1, 0]]

    return images


def find_transform_by_name(cfg, name):
    # Returns True if a transform name exists in augmentation cfg dict
    if isinstance(cfg, dict):
        if 'name' in cfg.keys() and cfg['name'] == name:
            return True
        else:
            if 'transforms' in cfg.keys() and isinstance(cfg['transforms'], (list, tuple)):
                return any([find_transform_by_name(t, name) for t in cfg['transforms']])
            return False


def get_transform_attr_by_name(cfg, name, attr):
    # Returns attr of first found transform by name in augmentation cfg dict
    if isinstance(cfg, dict):
        if 'name' in cfg.keys() and cfg['name'] == name:
            return cfg.get(attr)
        else:
            if 'transforms' in cfg.keys() and isinstance(cfg['transforms'], (list, tuple)):
                res = None
                for t in cfg['transforms']:
                    res = get_transform_attr_by_name(t, name, attr)
                    if res is not None:
                        break
                return res
            return None
