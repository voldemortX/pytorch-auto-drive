import torch.nn as nn
from ...common_models.blocks import DilatedBottleneck
from ...builder import MODELS


@MODELS.register()
def predefined_dilated_blocks(in_channels, mid_channels, dilations):
    # As in YOLOF
    blocks = [DilatedBottleneck(in_channels, mid_channels, dilation=d)
                for d in dilations]

    return nn.Sequential(*blocks)
