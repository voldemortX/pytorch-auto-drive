from torch import nn as nn
from torch.nn import functional as F

from ...builder import MODELS


@MODELS.register()
class SCNNDecoder(nn.Module):
    # Unused
    # SCNN original decoder for ResNet-101, very large channels, maybe impossible to add anything
    # resnet-101 -> H x W x 2048
    # 3x3 Conv -> H x W x 512
    # Dropout 0.1
    # 1x1 Conv -> H x W x 5
    # https://github.com/XingangPan/SCNN/issues/35

    def __init__(self, in_channels=2048, num_classes=5):
        super(SCNNDecoder, self).__init__()
        out_channels = in_channels // 4
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.dropout1 = nn.Dropout2d(0.1)
        self.conv2 = nn.Conv2d(out_channels, num_classes, 1, bias=False)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout1(x)
        x = self.conv2(x)

        return x
