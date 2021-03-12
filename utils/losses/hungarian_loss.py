import torch
from torch import Tensor
from typing import Optional
from torch.nn import functional as F
from ._utils import WeightedLoss


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
