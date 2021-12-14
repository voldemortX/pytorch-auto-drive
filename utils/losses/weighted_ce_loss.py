import torch.nn.functional as F

from ._utils import WeightedLoss
from .builder import LOSSES


# Use a base class that can take list weight instead of Tensor
@LOSSES.register()
class WeightedCrossEntropyLoss(WeightedLoss):
    __constants__ = ['ignore_index', 'reduction']

    def __init__(self, weight=None, size_average=None, ignore_index=-100,
                 reduce=None, reduction='mean'):
        super().__init__(weight, size_average, reduce, reduction)
        self.ignore_index = ignore_index

    def forward(self, inputs, targets):
        return F.cross_entropy(inputs, targets, weight=self.weight,
                               ignore_index=self.ignore_index, reduction=self.reduction)
