import torch.nn as nn

from ...builder import MODELS


@MODELS.register()
def predefined_dilated_blocks(in_channels, mid_channels, dilations):
    # As in YOLOF
    blocks = [MODELS.from_dict(
        dict(
            name='DilatedBottleneck',
            in_channels=in_channels,
            mid_channels=mid_channels,
            dilation=d
        )
    ) for d in dilations]

    return nn.Sequential(*blocks)
