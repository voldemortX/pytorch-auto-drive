import torch
from torch import Tensor
from typing import Optional
from torch.nn import functional as F
from ._utils import WeightedLoss


# Typical lane detection loss by binary segmentation (e.g. SCNN)
class LaneLoss(WeightedLoss):
    __constants__ = ['ignore_index', 'reduction']
    ignore_index: int

    def __init__(self, existence_weight: int = 0.1, weight: Optional[Tensor] = None, size_average=None,
                 ignore_index: int = -100, reduce=None, reduction: str = 'mean') -> None:
        super(LaneLoss, self).__init__(weight, size_average, reduce, reduction)
        self.ignore_index = ignore_index
        self.existence_weight = existence_weight

    def forward(self, inputs: Tensor, targets: Tensor, lane_existence: Tensor, net, interp_size) -> Tensor:
        outputs = net(inputs)
        prob_maps = torch.nn.functional.interpolate(outputs['out'], size=interp_size, mode='bilinear',
                                                    align_corners=True)
        targets[targets > lane_existence.shape[-1]] = 255  # Ignore extra lanes
        segmentation_loss = F.cross_entropy(prob_maps, targets, weight=self.weight,
                                            ignore_index=self.ignore_index, reduction=self.reduction)
        existence_loss = F.binary_cross_entropy_with_logits(outputs['lane'], lane_existence,
                                                            weight=None, pos_weight=None, reduction=self.reduction)
        total_loss = segmentation_loss + self.existence_weight * existence_loss

        return total_loss


# Loss function for SAD
class SADLoss(WeightedLoss):
    __constants__ = ['ignore_index', 'reduction']
    ignore_index: int

    def __init__(self, existence_weight: int = 0.1, weight: Optional[Tensor] = None, size_average=None,
                 ignore_index: int = -100, reduce=None, reduction: str = 'mean') -> None:
        super(SADLoss, self).__init__(weight, size_average, reduce, reduction)
        self.ignore_index = ignore_index
        self.existence_weight = existence_weight

    def forward(self, inputs: Tensor, targets: Tensor) -> Tensor:
        pass
