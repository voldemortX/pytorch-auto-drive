# Implementation based on pytorch 1.6.0
from .lane_seg_loss import LaneLoss, SADLoss
from .hungarian_loss import HungarianLoss
from .hungarian_bezier_loss import HungarianBezierLoss
from .weighted_ce_loss import WeightedCrossEntropyLoss
from .torch_loss import torch_loss
from .builder import LOSSES
