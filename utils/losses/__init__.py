# Implementation based on pytorch 1.6.0
from .lane_seg_loss import LaneLoss, SADLoss
from .hungarian_loss import HungarianLoss
from .weighted_ce_loss import WeightedCrossEntropyLoss
from .torch_loss import torch_loss
from .center_lane_loss import CenterLaneLoss
from .builder import LOSSES
