# Copied and modified from facebookresearch/detr and liuruijin17/LSTR
# mainly got rid of special types and argparse, added comments
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# Hungarian loss for LSTR

import torch
from torch import Tensor
from typing import Optional
from torch.nn import functional as F
from scipy.optimize import linear_sum_assignment
from ._utils import WeightedLoss


# TODO: Speed-up Hungarian on GPU with tensors
class HungarianMatcher(torch.nn.Module):
    """This class computes an assignment between the targets and the predictions of the network

    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(self, weight_class: float = 1, weight_curve: float = 1,
                 weight_lower: float = 1, weight_upper: float = 1):
        super().__init__()
        self.weight_class = weight_class
        self.weight_curve = weight_curve
        self.weight_lower = weight_lower
        self.weight_upper = weight_upper

    @torch.no_grad()
    def forward(self, outputs, targets):
        # Compute the matrices for an entire batch
        bs, num_queries = outputs["logits"].shape[:2]
        out_prob = outputs["logits"].flatten(0, 1).softmax(-1)
        tgt_ids = torch.cat([v["labels"] for v in targets])

        # Compute the classification cost. Contrary to the loss, we don't use the NLL,
        # but approximate it in 1 - prob[target class].
        # Then 1 can be omitted due to it is only a constant.
        cost_class = -out_prob[:, tgt_ids]  # [bs x max_lanes, num_GT]

        # 20210327
        out_bbox = outputs["curves"]
        tgt_uppers = torch.cat([tgt[:, 2] for tgt in targets])
        tgt_lowers = torch.cat([tgt[:, 1] for tgt in targets])

        # # Compute the L1 cost between lowers and uppers
        cost_lower = torch.cdist(out_bbox[:, :, 0].view((-1, 1)), tgt_lowers.unsqueeze(-1), p=1)
        cost_upper = torch.cdist(out_bbox[:, :, 1].view((-1, 1)), tgt_uppers.unsqueeze(-1), p=1)

        # # Compute the poly cost
        tgt_points = torch.cat([tgt[:, 3:] for tgt in targets])  # 0~20 112
        tgt_xs = tgt_points[:, :tgt_points.shape[1] // 2]
        valid_xs = tgt_xs >= 0
        weights = (torch.sum(valid_xs, dtype=torch.float32) / torch.sum(valid_xs, dim=1, dtype=torch.float32))**0.5
        weights = weights / torch.max(weights)

        tgt_ys = tgt_points[:, tgt_points.shape[1] // 2:]
        out_polys = out_bbox[:, :, 2:].view((-1, 6))
        tgt_ys = tgt_ys.repeat(out_polys.shape[0], 1, 1)
        tgt_ys = tgt_ys.transpose(0, 2)
        tgt_ys = tgt_ys.transpose(0, 1)

        # Calculate the predicted xs
        out_xs = out_polys[:, 0] / (tgt_ys - out_polys[:, 1]) ** 2 + out_polys[:, 2] / (tgt_ys - out_polys[:, 1]) + \
                 out_polys[:, 3] + out_polys[:, 4] * tgt_ys - out_polys[:, 5]
        tgt_xs = tgt_xs.repeat(out_polys.shape[0], 1, 1)
        tgt_xs = tgt_xs.transpose(0, 2)
        tgt_xs = tgt_xs.transpose(0, 1)

        cost_polys = torch.stack([torch.sum(torch.abs(tgt_x[valid_x] - out_x[valid_x]), dim=0) for tgt_x, out_x, valid_x in zip(tgt_xs, out_xs, valid_xs)], dim=-1)
        cost_polys = cost_polys * weights

        # # Final cost matrix
        C = self.weight_class * cost_class + self.weight_curve * cost_polys + \
            self.weight_lower * cost_lower + self.weight_upper * cost_upper

        C = C.view(bs, num_queries, -1).cpu()

        sizes = [tgt.shape[0] for tgt in targets]

        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]
        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]


# The Hungarian loss for LSTR
class HungarianLoss(WeightedLoss):
    __constants__ = ['ignore_index', 'reduction']
    ignore_index: int

    def __init__(self, weight: Optional[Tensor] = None, size_average=None,
                 ignore_index: int = -100, reduce=None, reduction: str = 'mean') -> None:
        super(HungarianLoss, self).__init__(weight, size_average, reduce, reduction)
        self.ignore_index = ignore_index

    def forward(self, inputs: Tensor, targets: Tensor, net) -> Tensor:
        outputs = net(inputs)
        # Match

        # Loss
        pass

    def curve_loss(self, inputs: Tensor, targets: Tensor) -> Tensor:
        # L1 loss on sample points, shouldn't it be direct regression?
        pass

    def vertical_loss(self, upper: Tensor, lower: Tensor, upper_target: Tensor, lower_target: Tensor) -> Tensor:
        # L1 loss on vertical start & end point,
        # corresponds to loss_lowers and loss_uppers in original LSTR code
        pass

    def classification_loss(self, inputs: Tensor, targets: Tensor) -> Tensor:
        # Typical classification loss (2 classes for lane detection)
        pass
