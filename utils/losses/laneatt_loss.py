import torch
from torch import Tensor
from typing import Optional
# from torch.nn import functional as F

from ._utils import WeightedLoss
from .builder import LOSSES
from .focal_loss import FocalLoss

INFINITY = 987654.


@LOSSES.register()
class LaneAttLoss(WeightedLoss):
    def __init__(self,
                 cls_weight: float = 10.,
                 reg_weight: float = 1.,
                 alpha: float = 0.25,
                 gamma: float = 2.,
                 num_strips: int = 72 - 1,
                 num_offsets: int = 72,
                 t_pos: float = 15.,
                 t_neg: float = 20.,
                 weight: Optional[Tensor] = None,
                 size_average=None,
                 ignore_index: int = -100,
                 reduce=None,
                 reduction: str = 'mean'):
        super(LaneAttLoss, self).__init__(weight, size_average, reduce, reduction)
        self.cls_weight = cls_weight
        self.reg_weight = reg_weight
        self.num_strips = num_strips
        self.num_offsets = num_offsets
        self.t_pos = t_pos
        self.t_neg = t_neg
        self.focal_loss = FocalLoss(alpha=alpha, gamma=gamma)
        self.smooth_l1_loss = torch.nn.SmoothL1Loss()

    def forward(self, inputs, targets, net):
        # inputs: batchsize x 3 x img_h x img_w
        # targets['offsets']: batchsize x max_lanes x (2 + 2 + 1 + num_offsets)
        targets = torch.stack([i['offsets'] for i in targets], dim=0)
        batch_size = inputs.shape[0]
        outputs = net(inputs)
        anchors_list = net.anchors.clone()
        anchors_list = anchors_list.repeat(batch_size, 1, 1)
        proposals_list = outputs['proposals_list']
        cls_loss = 0
        reg_loss = 0
        valid_imgs = len(targets)
        total_positives = 0
        for i in range(batch_size):
            anchors = anchors_list[i]
            proposals = proposals_list[i]
            target = targets[i]
            # Filter lanes that do not exist (confidence == 0)
            target = target[target[:, 1] == 1]
            if len(target) == 0:
                # If there are no targets, all proposals have to be negatives (i.e., 0 confidence)
                cls_target = proposals.new_zeros(len(proposals)).long()
                cls_pred = proposals[:, :2]
                cls_loss = cls_loss + self.focal_loss(cls_pred, cls_target).sum()
                continue

            # Gradients are also not necessary for the positive & negative matching
            with torch.no_grad():
                positives_mask, invalid_offsets_mask, negatives_mask, target_positives_indices = \
                    self.match_proposals_with_targets(anchors, target)
            # print("proposals.shape: {}, anchors.shape: {}, target: {}".format(
            #     proposals.shape, anchors.shape, target.shape))

            positives = proposals[positives_mask]
            num_positives = len(positives)
            total_positives += num_positives
            negatives = proposals[negatives_mask]
            num_negatives = len(negatives)
            # Handle edge case of no positives found
            if num_positives == 0:
                cls_target = proposals.new_zeros(len(proposals)).long()
                cls_pred = proposals[:, :2]
                cls_loss = cls_loss + self.focal_loss(cls_pred, cls_target).sum()
                continue

            # Get classification targets
            all_proposals = torch.cat([positives, negatives], 0)
            cls_target = proposals.new_zeros(num_positives + num_negatives).long()
            cls_target[:num_positives] = 1.
            cls_pred = all_proposals[:, :2]

            # Regression targets
            reg_pred = positives[:, 4:]
            with torch.no_grad():
                target = target[target_positives_indices]
                positive_starts = (positives[:, 2] * self.num_strips).round().long()
                targets_starts = (target[:, 2] * self.num_strips).round().long()
                # targets_starts = (target[:, 2] * self.num_strips)
                target[:, 4] -= positive_starts - targets_starts
                all_indices = torch.arange(num_positives, dtype=torch.long)
                ends = (positive_starts + target[:, 4] - 1).round().long()
                # ends = (positive_starts + target[:, 4] - 1)
                # length + num_offsets + pad (assignment trick ?)
                invalid_offsets_mask = torch.zeros((num_positives, 1 + self.num_offsets + 1), dtype=torch.int)
                invalid_offsets_mask[all_indices, 1 + positive_starts] = 1
                invalid_offsets_mask[all_indices, 1 + ends + 1] -= 1
                invalid_offsets_mask = invalid_offsets_mask.cumsum(dim=1) == 0
                invalid_offsets_mask = invalid_offsets_mask[:, :-1]
                invalid_offsets_mask[:, 0] = False
                # ? ? ?
                # target.dtype = torch.long
                reg_target = target[:, 4:].float()
                reg_target[invalid_offsets_mask] = reg_pred[invalid_offsets_mask]

            # loss
            reg_loss += self.smooth_l1_loss(reg_pred, reg_target)
            cls_loss += self.focal_loss(cls_pred, cls_target).sum() / num_positives

        # Batch mean
        reg_loss /= valid_imgs
        cls_loss /= valid_imgs

        total_loss = self.cls_weight * cls_loss + self.reg_weight * reg_loss

        return total_loss, {'total loss': total_loss, 'cls loss': cls_loss, 'reg loss': reg_loss,
                            'batch_positives': total_positives}

    def match_proposals_with_targets(self, proposals, targets):
        # repeat proposals and targets to generate all combinations
        num_proposals = proposals.shape[0]
        num_targets = targets.shape[0]

        # pad proposals and target for the vaild_offset_mask' s trick
        # ? ? ?
        proposals_pad = proposals.new_zeros(proposals.shape[0], proposals.shape[1] + 1)
        proposals_pad[:, :-1] = proposals
        proposals = proposals_pad
        targets_pad = targets.new_zeros(targets.shape[0], targets.shape[1] + 1)
        targets_pad[:, :-1] = targets
        targets = targets_pad

        # repeat interleave'ing [a, b] 2 times gives [a, a, b, b]
        proposals = torch.repeat_interleave(proposals, num_targets, dim=0)
        # applying this 2 times on [c, d] gives [c, d, c, d]
        targets = torch.cat(num_proposals * [targets])

        # get start and the the intersection of offsets
        # ? ? ?
        targets_starts = targets[:, 2] * self.num_strips
        proposals_starts = proposals[:, 2] * self.num_strips
        # print(targets_starts.sum())
        # print(proposals_starts)
        starts = torch.max(targets_starts.float(), proposals_starts).round().long()
        ends = (targets_starts + targets[:, 4].float() - 1.).round().long()
        lengths = ends - starts + 1
        ends[lengths < 0] = starts[lengths < 0] - 1
        lengths[lengths < 0] = 0  # a negative number here means no intersection, thus no length

        # generate valid offsets mask, which works like this:
        #  start with mask [0, 0, 0, 0, 0]
        #  suppose start = 1
        #  lenght = 2
        valid_offsets_mask = targets.new_zeros(targets.shape)
        all_indices = torch.arange(valid_offsets_mask.shape[0], dtype=torch.long, device=targets.device)
        #  put a one on index `start`, giving [0, 1, 0, 0, 0]
        valid_offsets_mask[all_indices, 5 + starts] = 1
        #  put a -1 on the `end` index, giving [0, 1, 0, -1, 0]
        valid_offsets_mask[all_indices, 5 + ends + 1] -= 1
        valid_offsets_mask = valid_offsets_mask.cumsum(dim=1) != 0
        invalid_offsets_mask = ~valid_offsets_mask

        # compute distance
        # this compares [ac, ad, bc, bd], i.e., all combinations
        distances = torch.abs((targets - proposals) * valid_offsets_mask.float()).sum(dim=1) / \
                    (lengths.float() + 1e-9)  # avoid division by zero
        distances[lengths == 0] = INFINITY
        invalid_offsets_mask = invalid_offsets_mask.view(num_proposals, num_targets, invalid_offsets_mask.shape[1])
        distances = distances.view(num_proposals, num_targets)  # d[i,j] = distance from proposal i to target j

        positives = distances.min(dim=1)[0] < self.t_pos
        negatives = distances.min(dim=1)[0] > self.t_neg

        if positives.sum() == 0:
            target_positives_indices = torch.tensor([], device=positives.device, dtype=torch.long)
        else:
            target_positives_indices = distances[positives].argmin(dim=1)

        invalid_offsets_mask = invalid_offsets_mask[positives, target_positives_indices]
        return positives, invalid_offsets_mask[:, :-1], negatives, target_positives_indices
