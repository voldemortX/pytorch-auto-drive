# Based on torchvision
from .builder import MODELS
from . import segmentation
from . import lane_detection
from . import transformer
from . import common_models
from .backbone_wrappers import VGG16, free_resnet_backbone, predefined_resnet_backbone
from .mobilenet_v2 import MobileNetV2Encoder
from .mobilenet_v3 import MobileNetV3Encoder
from .rep_vgg import RepVggEncoder
