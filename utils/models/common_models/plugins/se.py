from torch import nn as nn

from ..._utils import make_divisible


class SELayer(nn.Module):
    """Squeeze-and-Excitation Module.
    Args:
        channels (int): The input (and output) channels of the SE layer.
        ratio (int): Squeeze ratio in SELayer, the intermediate channel will be
            ``int(channels/ratio)``. Default: 16.
    """

    def __init__(self, channels, ratio=16, act=nn.ReLU, scale_act=nn.Sigmoid):
        super(SELayer, self).__init__()

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(in_channels=channels, out_channels=make_divisible(channels // ratio, 8), kernel_size=1,
                             stride=1)
        self.fc2 = nn.Conv2d(in_channels=make_divisible(channels // ratio, 8), out_channels=channels, kernel_size=1,
                             stride=1)
        self.activation = act()
        self.scale_activation = scale_act()

    def forward(self, x):
        out = self.avgpool(x)
        out = self.fc1(out)
        out = self.activation(out)
        out = self.fc2(out)
        out = self.scale_activation(out)

        return x * out
