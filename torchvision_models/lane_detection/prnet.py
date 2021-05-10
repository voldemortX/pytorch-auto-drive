# Implementation of Polynomial Regression Network based on the original paper (PRNet):
# http://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123630698.pdf
import torch.nn as nn
from collections import OrderedDict
from common_models import RESAReducer, SCNN_D
from .. import resnet
from ..segmentation import erfnet_resnet
from .._utils import IntermediateLayerGetter


# One convolution layer for each branch
# The kernel size 3x3 is an educated guess, the 3 branches are implemented separately for future flexibility
class PolynomialBranch(nn.Module):
    def __init__(self, in_channels, order=2):
        super(PolynomialBranch, self).__init__()
        self.conv = nn.Conv2d(in_channels, order + 1, kernel_size=3, stride=1, padding=1, bias=False)

    def forward(self, inputs):
        return self.conv(inputs)


class InitializationBranch(nn.Module):
    def __init__(self, in_channels):
        super(InitializationBranch, self).__init__()
        self.conv = nn.Conv2d(in_channels, 1, kernel_size=3, stride=1, padding=1, bias=False)

    def forward(self, inputs):
        return self.conv(inputs)


class HeightBranch(nn.Module):
    def __init__(self, in_channels):
        super(HeightBranch, self).__init__()
        self.conv = nn.Conv2d(in_channels, 1, kernel_size=3, stride=1, padding=1, bias=False)

    def forward(self, inputs):
        return self.conv(inputs)


# Currently supported backbones: ERFNet, ResNets
class PRNet(nn.Module):
    def __init__(self, backbone_name, dropout_1=0.3, dropout_2=0.03, order=2):
        super(PRNet, self).__init__()
        if backbone_name == 'erfnet':
            self.backbone = erfnet_resnet(dropout_1=dropout_1, dropout_2=dropout_2, encoder_only=True)
            in_channels = 128
        else:
            in_channels = 2048 if backbone_name == 'resnet50' or backbone_name == 'resnet101' else 512
            backbone = resnet.__dict__[backbone_name](
                pretrained=True,
                replace_stride_with_dilation=[False, True, True])
            return_layers = {'layer4': 'out'}
            self.backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)

        self.channel_reducer = RESAReducer(in_channels=in_channels)
        self.spatial_conv = SCNN_D()
        self.polynomial_branch = PolynomialBranch(in_channels=128, order=order)
        self.initialization_branch = InitializationBranch(in_channels=128)
        self.height_branch = HeightBranch(in_channels=128)

    def forward(self, inputs):
        # Encoder (8x down-sampling) -> channel reduction (128, another educated guess) -> SCNN_D -> 3 branches
        out = OrderedDict()
        x = self.backbone(inputs)
        x = self.channel_reducer(x)
        x = self.spatial_conv(x)
        out['polynomials'] = self.polynomial_branch(x)
        out['initializations'] = self.initialization_branch(x)
        out['heights'] = self.height_branch(x)

        return out
