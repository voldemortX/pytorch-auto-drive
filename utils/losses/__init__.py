# Implementation based on pytorch 1.6.0
from .lane_seg_loss import LaneLoss, SADLoss
from .hungarian_loss import HungarianLoss
from .weighted_ce_loss import WeightedCrossEntropyLoss
from .torch_loss import torch_loss
from .focal_loss import _focal_loss, FocalLoss
from .laneatt_loss import LaneAttLoss
from .builder import LOSSES
