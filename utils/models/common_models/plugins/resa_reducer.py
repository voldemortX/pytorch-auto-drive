from torch import nn as nn
from torch.nn import functional as F

from ...builder import MODELS


@MODELS.register()
class RESAReducer(nn.Module):
    # Reduce channel (typically to 128), RESA code use no BN nor ReLU

    def __init__(self, in_channels=512, reduce=128, bn_relu=True):
        super(RESAReducer, self).__init__()
        self.bn_relu = bn_relu
        self.conv1 = nn.Conv2d(in_channels, reduce, 1, bias=False)
        if self.bn_relu:
            self.bn1 = nn.BatchNorm2d(reduce)

    def forward(self, x):
        x = self.conv1(x)
        if self.bn_relu:
            x = self.bn1(x)
            x = F.relu(x)

        return x
