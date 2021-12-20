from torch import nn

from ..builder import MODELS


@MODELS.register()
class FCNHead(nn.Sequential):
    def __init__(self, in_channels, num_classes):
        inter_channels = in_channels // 4
        layers = [
            nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(inter_channels, num_classes, 1)
        ]

        super(FCNHead, self).__init__(*layers)
