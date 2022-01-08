from torch import nn as nn

from ...builder import MODELS


@MODELS.register()
class PlainDecoder(nn.Module):
    # Plain decoder (albeit simplest) from the RESA paper

    def __init__(self, in_channels=128, num_classes=5):
        super(PlainDecoder, self).__init__()
        self.dropout1 = nn.Dropout2d(0.1)
        self.conv1 = nn.Conv2d(in_channels, num_classes, 1, bias=True)

    def forward(self, x):
        x = self.dropout1(x)
        x = self.conv1(x)

        return x
