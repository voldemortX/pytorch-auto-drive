# Implemented upon pytorch 1.6.0
import torch
from torch import Tensor
from typing import Optional
from torch.nn import _reduction as _Reduction
from torch.nn import functional as F


class _Loss(torch.nn.Module):
    reduction: str

    def __init__(self, size_average=None, reduce=None, reduction: str = 'mean') -> None:
        super(_Loss, self).__init__()
        if size_average is not None or reduce is not None:
            self.reduction = _Reduction.legacy_get_string(size_average, reduce)
        else:
            self.reduction = reduction


class _WeightedLoss(_Loss):
    def __init__(self, weight: Optional[Tensor] = None, size_average=None, reduce=None,
                 reduction: str = 'mean') -> None:
        super(_WeightedLoss, self).__init__(size_average, reduce, reduction)
        self.register_buffer('weight', weight)


# Typical lane detection loss by binary segmentation (e.g. SCNN)
class LaneLoss(_WeightedLoss):
    __constants__ = ['ignore_index', 'reduction']
    ignore_index: int

    def __init__(self, existence_weight: int = 1, weight: Optional[Tensor] = None, size_average=None,
                 ignore_index: int = -100, reduce=None, reduction: str = 'mean') -> None:
        super(LaneLoss, self).__init__(weight, size_average, reduce, reduction)
        self.ignore_index = ignore_index
        self.existence_weight = existence_weight

    def forward(self, inputs: Tensor, targets: Tensor, lane_existence: Tensor, net) -> Tensor:
        outputs = net(inputs)
        segmentation_loss = F.cross_entropy(outputs['out'], targets, weight=self.weight,
                                            ignore_index=self.ignore_index, reduction=self.reduction)
        existence_loss = F.cross_entropy(outputs['aux'], lane_existence, weight=None,
                                         ignore_index=-100, reduction=self.reduction)
        total_loss = segmentation_loss + self.existence_weight * existence_loss

        return total_loss


# Loss function for SAD
class SADLoss(_WeightedLoss):
    __constants__ = ['ignore_index', 'reduction']
    ignore_index: int

    def __init__(self, weight: Optional[Tensor] = None, size_average=None, ignore_index: int = -100,
                 reduce=None, reduction: str = 'mean') -> None:
        super(SADLoss, self).__init__(weight, size_average, reduce, reduction)
        pass

    def forward(self, inputs: Tensor, targets: Tensor) -> Tensor:
        pass
