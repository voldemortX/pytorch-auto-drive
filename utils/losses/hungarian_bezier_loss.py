# Copied and modified from facebookresearch/detr
# Refactored and added comments


import torch
from torch.nn import functional as F
from scipy.optimize import linear_sum_assignment

from ..ddp_utils import is_dist_avail_and_initialized, get_world_size
from ..curve_utils import BezierSampler, cubic_bezier_curve_segment, get_valid_points
from ._utils import WeightedLoss
from .hungarian_loss import HungarianLoss
from .builder import LOSSES


# TODO: Speed-up Hungarian on GPU with tensors
class _HungarianMatcher(torch.nn.Module):
    """This class computes an assignment between the targets and the predictions of the network

    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    POTO matching, which maximizes the cost matrix.
    """

    def __init__(self, alpha=0.8, bezier_order=3, num_sample_points=100, k=7):
        super().__init__()
        self.k = k
        self.alpha = alpha
        self.num_sample_points = num_sample_points
        self.bezier_sampler = BezierSampler(num_sample_points=num_sample_points, order=bezier_order)

    @torch.no_grad()
    def forward(self, outputs, targets):
        # Compute the matrices for an entire batch (computation is all pairs, in a way includes the real loss function)
        # targets: each target: ['keypoints': L x N x 2]
        # B: batch size; Q: max lanes per-pred, G: total num ground-truth-lanes
        B, Q = outputs["logits"].shape
        target_keypoints = torch.cat([i['keypoints'] for i in targets], dim=0)  # G x N x 2
        target_sample_points = torch.cat([i['sample_points'] for i in targets], dim=0)  # G x num_sample_points x 2

        # Valid bezier segments
        target_keypoints = cubic_bezier_curve_segment(target_keypoints, target_sample_points)
        target_sample_points = self.bezier_sampler.get_sample_points(target_keypoints)

        # target_valid_points = get_valid_points(target_sample_points)  # G x num_sample_points
        G, N = target_keypoints.shape[:2]
        out_prob = outputs["logits"].sigmoid()  # B x Q
        out_lane = outputs['curves']  # B x Q x N x 2
        sizes = [target['keypoints'].shape[0] for target in targets]

        # 1. Local maxima prior
        _, max_indices = torch.nn.functional.max_pool1d(out_prob.unsqueeze(1),
                                                        kernel_size=self.k, stride=1,
                                                        padding=(self.k - 1) // 2, return_indices=True)
        max_indices = max_indices.squeeze(1)  # B x Q
        indices = torch.arange(0, Q, dtype=out_prob.dtype, device=out_prob.device).unsqueeze(0).expand_as(max_indices)
        local_maxima = (max_indices == indices).flatten().unsqueeze(-1).expand(-1, G)  # BQ x G

        # Safe reshape
        out_prob = out_prob.flatten()  # BQ
        out_lane = out_lane.flatten(end_dim=1)  # BQ x N x 2

        # 2. Compute the classification cost. Contrary to the loss, we don't use the NLL,
        # but approximate it in 1 - prob[target class].
        # Then 1 can be omitted due to it is only a constant.
        # For binary classification, it is just prob (understand this prob as objectiveness in OD)
        cost_label = out_prob.unsqueeze(-1).expand(-1, G)  # BQ x G

        # 3. Compute the curve sampling cost
        cost_curve = 1 - torch.cdist(self.bezier_sampler.get_sample_points(out_lane).flatten(start_dim=-2),
                                     target_sample_points.flatten(start_dim=-2),
                                     p=1) / self.num_sample_points  # BQ x G

        # Bound the cost to [0, 1]
        cost_curve = cost_curve.clamp(min=0, max=1)

        # Final cost matrix (scipy uses min instead of max)
        C = local_maxima * cost_label ** (1 - self.alpha) * cost_curve ** self.alpha
        C = -C.view(B, Q, -1).cpu()

        # Hungarian (weighted) on each image
        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]

        # Return (pred_indices, target_indices) for each image
        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]


@LOSSES.register()
class HungarianBezierLoss(WeightedLoss):
    def __init__(self, curve_weight=1, label_weight=0.1, seg_weight=0.75, alpha=0.8,
                 num_sample_points=100, bezier_order=3, weight=None, size_average=None, reduce=None, reduction='mean',
                 ignore_index=-100, weight_seg=None, k=9):
        super().__init__(weight, size_average, reduce, reduction)
        self.curve_weight = curve_weight  # Weight for sampled points' L1 distance error between curves
        self.label_weight = label_weight  # Weight for classification error
        self.seg_weight = seg_weight  # Weight for binary segmentation auxiliary task
        self.weight_seg = weight_seg  # BCE loss weight
        self.ignore_index = ignore_index
        self.bezier_sampler = BezierSampler(num_sample_points=num_sample_points, order=bezier_order)
        self.matcher = _HungarianMatcher(alpha=alpha, num_sample_points=num_sample_points, bezier_order=bezier_order,
                                         k=k)
        if self.weight is not None and not isinstance(self.weight, torch.Tensor):
            self.weight = torch.tensor(self.weight).cuda()
        if self.weight_seg is not None and not isinstance(self.weight_seg, torch.Tensor):
            self.weight_seg = torch.tensor(self.weight_seg).cuda()
        self.register_buffer('pos_weight', self.weight[1] / self.weight[0])
        self.register_buffer('pos_weight_seg', self.weight_seg[1] / self.weight_seg[0])

    def forward(self, inputs, targets, net):
        outputs = net(inputs)
        output_curves = outputs['curves']
        target_labels = torch.zeros_like(outputs['logits'])
        target_segmentations = torch.stack([target['segmentation_mask'] for target in targets])

        total_targets = 0
        for i in targets:
            total_targets += i['keypoints'].numel()

        # CULane actually can produce a whole batch of no-lane images,
        # in which case, we just calculate the classification loss
        if total_targets > 0:
            # Match
            indices = self.matcher(outputs=outputs, targets=targets)
            idx = HungarianLoss.get_src_permutation_idx(indices)
            output_curves = output_curves[idx]

            # Targets (rearrange each lane in the whole batch)
            # B x N x ... -> BN x ...
            target_keypoints = torch.cat([t['keypoints'][i] for t, (_, i) in zip(targets, indices)], dim=0)
            target_sample_points = torch.cat([t['sample_points'][i] for t, (_, i) in zip(targets, indices)], dim=0)

            # Valid bezier segments
            target_keypoints = cubic_bezier_curve_segment(target_keypoints, target_sample_points)
            target_sample_points = self.bezier_sampler.get_sample_points(target_keypoints)

            target_labels[idx] = 1  # Any matched lane has the same label 1

        else:
            # For DDP
            target_sample_points = torch.tensor([], dtype=torch.float32, device=output_curves.device)

        target_valid_points = get_valid_points(target_sample_points)
        # Loss
        loss_curve = self.point_loss(self.bezier_sampler.get_sample_points(output_curves),
                                     target_sample_points)
        loss_label = self.classification_loss(inputs=outputs['logits'], targets=target_labels)
        loss_seg = self.binary_seg_loss(inputs=outputs['segmentations'], targets=target_segmentations)

        loss = self.label_weight * loss_label + self.curve_weight * loss_curve + self.seg_weight * loss_seg

        return loss, {'training loss': loss, 'loss label': loss_label, 'loss curve': loss_curve,
                      'loss seg': loss_seg,
                      'valid portion': target_valid_points.float().mean()}

    def point_loss(self, inputs, targets, valid_points=None):
        # L1 loss on sample points
        # inputs/targets: L x N x 2
        # valid points: L x N
        if targets.numel() == 0:
            targets = inputs.clone().detach()
        loss = F.l1_loss(inputs, targets, reduction='none')
        if valid_points is not None:
            loss *= valid_points.unsqueeze(-1)
            normalizer = valid_points.sum()
        else:
            normalizer = targets.shape[0] * targets.shape[1]
            normalizer = torch.as_tensor([normalizer], dtype=inputs.dtype, device=inputs.device)
        if self.reduction == 'mean':
            if is_dist_avail_and_initialized():  # Global normalizer should be same across devices
                torch.distributed.all_reduce(normalizer)
            normalizer = torch.clamp(normalizer / get_world_size(), min=1).item()
            loss = loss.sum() / normalizer
        elif self.reduction == 'sum':  # Usually not needed, but let's have it anyway
            loss = loss.sum()

        return loss

    def classification_loss(self, inputs, targets):
        # Typical classification loss (cross entropy)
        # No need for permutation, assume target is matched to inputs

        # Negative weight as positive weight
        return F.binary_cross_entropy_with_logits(inputs.unsqueeze(1), targets.unsqueeze(1), pos_weight=self.pos_weight,
                                                  reduction=self.reduction) / self.pos_weight

    def binary_seg_loss(self, inputs, targets):
        # BCE segmentation loss with weighting and ignore index
        # No relation whatever to matching

        # Process inputs
        inputs = torch.nn.functional.interpolate(inputs, size=targets.shape[-2:], mode='bilinear', align_corners=True)
        inputs = inputs.squeeze(1)

        # Process targets
        valid_map = (targets != self.ignore_index)
        targets[~valid_map] = 0
        targets = targets.float()

        # Negative weight as positive weight
        loss = F.binary_cross_entropy_with_logits(inputs, targets, pos_weight=self.pos_weight_seg,
                                                  reduction='none') / self.pos_weight_seg
        loss *= valid_map

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss
