import torch
from torch import Tensor
from typing import Optional
from torch.nn import _reduction as _Reduction


class _Loss(torch.nn.Module):
    reduction: str

    def __init__(self, size_average=None, reduce=None, reduction: str = 'mean') -> None:
        super(_Loss, self).__init__()
        if size_average is not None or reduce is not None:
            self.reduction = _Reduction.legacy_get_string(size_average, reduce)
        else:
            self.reduction = reduction


class WeightedLoss(_Loss):
    def __init__(self, weight: Optional[Tensor] = None, size_average=None, reduce=None,
                 reduction: str = 'mean') -> None:
        super(WeightedLoss, self).__init__(size_average, reduce, reduction)
        if not isinstance(weight, Tensor):
            weight = torch.tensor(weight).cuda()
        self.register_buffer('weight', weight)
