# Some codes may look like copy-paste, but every line of code is manually checked
# Implementation reference order (not strict):
# Paper Content = Official Impl > TorchVision Impl > OpenMMLab Impl > Community Impl
# If there exists an official improved code version that is different from paper content,
# probably it will not be used here in PytorchAutoDrive.
from .builder import MODELS
from . import segmentation
from . import common_models
from .backbone_wrappers import free_resnet_backbone, predefined_resnet_backbone
