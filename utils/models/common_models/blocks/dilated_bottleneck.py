import torch
import torch.nn as nn

from ...builder import MODELS


@MODELS.register()
class DilatedBottleneck(nn.Module):
    # Refactored from https://github.com/chensnathan/YOLOF/blob/master/yolof/modeling/encoder.py
    # Diff from typical ResNetV1.5 BottleNeck:
    # flexible expansion rate, forbids downsampling, relu immediately after last conv
    def __init__(self,
                 in_channels=512,
                 mid_channels=128,
                 dilation=1):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=1, padding=0),
            nn.BatchNorm2d(mid_channels)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(mid_channels, mid_channels,
                      kernel_size=3, padding=dilation, dilation=dilation),
            nn.BatchNorm2d(mid_channels)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(mid_channels, in_channels, kernel_size=1, padding=0),
            nn.BatchNorm2d(in_channels)
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.relu(out)
        out = out + identity

        return out
