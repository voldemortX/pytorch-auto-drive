# Loss for PRNet
import torch
from torch.nn import functional as F
from ._utils import WeightedLoss


class PRLoss(WeightedLoss):
    __constants__ = ['reduction']
    ignore_index: int

    def __init__(self, polynomial_weight=1, initialization_weight=1, height_weight=0.1, beta=0.005, m=20,
                 weight=None, size_average=None, reduce=None, reduction='mean'):
        super(PRLoss, self).__init__(weight, size_average, reduce, reduction)
        self.polynomial_weight = polynomial_weight
        self.initialization_weight = initialization_weight
        self.height_weight = height_weight
        self.beta = beta  # Beta for smoothed L1 loss
        self.m = m  # Number of sample points to calculate polynomial regression loss

    def forward(self, inputs, targets, net):
        outputs = net(inputs)
        pass

    @staticmethod
    def beta_smoothed_l1_loss(inputs, targets, beta=0.005):
        # Smoothed L1 loss with a hyper-parameter (as in PRNet paper)
        # The original torch F.smooth_l1_loss() is equivalent to beta=1
        t = torch.abs(inputs - targets)

        return torch.where(t < beta, 0.5 * t ** 2 / beta, t - 0.5 * beta)
