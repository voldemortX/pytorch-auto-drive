# Implementation based on pytorch 1.6.0
from .lane_seg_loss import LaneLoss, SADLoss
from .hungarian_loss import HungarianLoss
from .builder import LOSSES
