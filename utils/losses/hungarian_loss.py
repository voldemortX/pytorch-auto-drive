# Copied and modified from facebookresearch/detr and liuruijin17/LSTR
# Refactored and added comments
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# Hungarian loss for LSTR

import torch
from torch import Tensor
from typing import Optional
from torch.nn import functional as F
from scipy.optimize import linear_sum_assignment
from ._utils import WeightedLoss


def lane_normalize_in_batch(keypoints):
    # Calculate normalization weights for lanes with different number of valid sample points,
    # so they can produce loss in a similar scale: rather weird but it is what LSTR did
    # https://github.com/liuruijin17/LSTR/blob/6044f7b2c5892dba7201c273ee632b4962350223/models/py_utils/matcher.py#L59
    # keypoints: [..., N, 2], ... means arbitrary number of leading dimensions
    valid_points = keypoints[..., 0] > 0
    norm_weights = (valid_points.sum().float() / valid_points.sum(dim=-1).float()) ** 0.5
    norm_weights /= norm_weights.max()

    return norm_weights, valid_points  # [...], [..., N]


def _cubic_curve_with_projection(coefficients, y):
    # The cubic curve model from LSTR (considers projection to image plane)
    # Return x coordinates
    # parameters: [..., 6], ... means arbitrary number of leading dimensions
    # 6 coefficients: [k", f", m", n", b", b''']
    # y: [..., N]
    x = coefficients[:, 0] / (y - coefficients[:, 1]) ** 2 \
        + coefficients[:, 2] / (y - coefficients[:, 1]) \
        + coefficients[:, 3] \
        + coefficients[:, 4] * y \
        - coefficients[:, 5]

    return x  # [..., N]


# TODO: Speed-up Hungarian on GPU with tensors
class HungarianMatcher(torch.nn.Module):
    """This class computes an assignment between the targets and the predictions of the network

    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(self, upper_weight=2, lower_weight=2, curve_weight=5, label_weight=3):
        super().__init__()
        self.lower_weight = lower_weight
        self.upper_weight = upper_weight
        self.curve_weight = curve_weight
        self.label_weight = label_weight

    @torch.no_grad()
    def forward(self, outputs, targets):
        # Compute the matrices for an entire batch (computation is all pairs, in a way includes the real loss function)
        # targets: each target: ['keypoints': L x N x 2, 'padding_mask': H x W, 'uppers': L, 'lowers': L, 'labels': L]
        # B: bs; Q: max lanes per-pred, L: num lanes, N: num keypoints per-lane, G: total num ground-truth-lanes
        bs, num_queries = outputs["logits"].shape[:2]
        out_prob = outputs["logits"].flatten(end_dim=-2).sigmoid()  # BQ x 1
        out_lane = outputs['curves'].flatten(end_dim=-2)  # BQ x 8
        target_uppers = torch.cat([i['uppers'] for i in targets])
        target_lowers = torch.cat([i['lowers'] for i in targets])
        sizes = [target['labels'].shape[0] for target in targets]
        num_gt = sum(sizes)

        # 1. Compute the classification cost. Contrary to the loss, we don't use the NLL,
        # but approximate it in 1 - prob[target class].
        # Then 1 can be omitted due to it is only a constant.
        # For binary classification, it is just prob (understand this prob as objectiveness in OD)
        cost_label = -out_prob.repeat(1, num_gt)  # BQ x G

        # 2. Compute the L1 cost between lowers and uppers
        cost_lower = torch.cdist(out_lane[:, 0], target_uppers.unsqueeze(-1), p=1)  # BQ x G
        cost_upper = torch.cdist(out_lane[:, 1], target_lowers.unsqueeze(-1), p=1)  # BQ x G

        # 3. Compute the curve cost
        target_keypoints = torch.cat([i['keypoints'] for i in targets], dim=0)  # G x N x 2
        norm_weights, valid_points = lane_normalize_in_batch(target_keypoints)  # G, G x N
        out_x = _cubic_curve_with_projection(coefficients=out_lane[:, 2:], y=target_keypoints[0, :, 1])  # BQ x N

        # Masked torch.cdist(p=1)
        expand_shape = [bs * num_queries, num_gt, out_x.shape[-1]]  # BQ x G x N
        cost_curve = ((out_x.unsqueeze(1).expand(expand_shape) -
                      target_keypoints[:, :, 0].unsqueeze(0).expand(expand_shape)).abs() *
                      valid_points.unsqueeze(0).expand(expand_shape)).sum(-1)  # BQ x G
        cost_curve *= norm_weights  # BQ x G

        # Final cost matrix
        C = self.label_weight * cost_label + self.curve_weight * cost_curve + \
            self.lower_weight * cost_lower + self.upper_weight * cost_upper
        C = C.view(bs, num_queries, -1).cpu()

        # Hungarian (weighted) on each image
        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]

        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]


# The Hungarian loss for LSTR
class HungarianLoss(WeightedLoss):
    __constants__ = ['reduction']

    def __init__(self, upper_weight=2, lower_weight=2, curve_weight=5, label_weight=3,
                 weight=None, size_average=None, reduce=None, reduction='mean'):
        super(HungarianLoss, self).__init__(weight, size_average, reduce, reduction)
        self.lower_weight = lower_weight
        self.upper_weight = upper_weight
        self.curve_weight = curve_weight
        self.label_weight = label_weight
        self.matcher = HungarianMatcher(upper_weight, lower_weight, curve_weight, label_weight)

    def forward(self, inputs: Tensor, targets: Tensor, net) -> Tensor:
        outputs = net(inputs)
        
        # Match
        indices = self.matcher(outputs=outputs, targets=targets)
        
        # Targets
        target_lowers = torch.cat([t['lowers'] for t in targets], dim=0)
        target_uppers = torch.cat([t['uppers'] for t in targets], dim=0)
        target_keypoints = torch.cat([t['keypoints'] for t in targets], dim=0)
        # target_labels = torch.cat([t['labels'] for t in targets], dim=0)
        
        # Loss

    def curve_loss(self, inputs: Tensor, targets: Tensor, indices: Tensor) -> Tensor:
        # L1 loss on sample points, shouldn't it be direct regression?
        pass

    def vertical_loss(self, inputs: Tensor, targets: Tensor, indices: Tensor) -> Tensor:
        # L1 loss on vertical start & end point,
        # corresponds to loss_lowers and loss_uppers in original LSTR code
        pass

    def classification_loss(self, inputs: Tensor, targets: Tensor, indices: Tensor) -> Tensor:
        # Typical classification loss (binary classification)
        
        return F.binary_cross_entropy_with_logits(inputs[indices], targets,
                                                  weight=None, pos_weight=None, reduction=self.reduction)
