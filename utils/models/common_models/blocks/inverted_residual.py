from torch import nn as nn

from ..plugins import SELayer


class InvertedResidual(nn.Module):
    """InvertedResidual block for MobileNetV2.
    Args:
        in_channels (int): The input channels of the InvertedResidual block.
        out_channels (int): The output channels of the InvertedResidual block.
        stride (int): Stride of the middle (first) 3x3 convolution.
        expand_ratio (int): Adjusts number of channels of the hidden layer
            in InvertedResidual by this amount.
        dilation (int): Dilation rate of depthwise conv. Default: 1
    Returns:
        Tensor: The output tensor.
    """

    def __init__(self, in_channels, out_channels, stride, expand_ratio, dilation=1, bias=False):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2], f'stride must in [1, 2]. ' \
                                 f'But received {stride}.'
        self.use_res_connect = self.stride == 1 and in_channels == out_channels
        hidden_dim = int(round(in_channels * expand_ratio))
        layers = []
        if expand_ratio != 1:
            layers.extend([
                nn.Conv2d(in_channels=in_channels, out_channels=hidden_dim, kernel_size=1, bias=bias),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6()  # min(max(0, x), 6)
            ])

        layers.extend([
            nn.Conv2d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=3, stride=stride,
                      padding=dilation, dilation=dilation, groups=hidden_dim, bias=bias),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(),
            nn.Conv2d(in_channels=hidden_dim, out_channels=out_channels, kernel_size=1, bias=bias),
            nn.BatchNorm2d(out_channels)
        ])
        self.conv = nn.Sequential(*layers)

    def forward(self, x):

        def _inner_forward(x):
            if self.use_res_connect:
                return x + self.conv(x)
            else:
                return self.conv(x)

        out = _inner_forward(x)

        return out


class InvertedResidualV3(nn.Module):
    """Inverted Residual Block for MobileNetV3.
    Args:
        in_channels (int): The input channels of this Module.
        out_channels (int): The output channels of this Module.
        mid_channels (int): The input channels of the depthwise convolution.
        kernel_size (int): The kernel size of the depthwise convolution. Default: 3.
        stride (int): The stride of the depthwise convolution. Default: 1.
        with_se (dict): with or without se layer. Default: False, which means no se layer.
        with_expand_conv (bool): Use expand conv or not. If set False,
            mid_channels must be the same with in_channels. Default: True.
        act_cfg (dict): Config dict for activation layer.
            Default: dict(type='ReLU').
    Returns:
        Tensor: The output tensor.
    """

    def __init__(self, in_channels, out_channels, mid_channels, kernel_size=3, stride=1, with_se=False,
                 with_expand_conv=True, act='HSwish', bias=False, dilation=1):
        super(InvertedResidualV3, self).__init__()
        self.with_res_shortcut = (stride == 1 and in_channels == out_channels)
        assert stride in [1, 2]
        activation_layer = nn.Hardswish if act == 'HSwish' else nn.ReLU6
        self.with_se = with_se
        self.with_expand_conv = with_expand_conv
        if not self.with_expand_conv:
            assert mid_channels == in_channels
        if self.with_expand_conv:
            self.expand_conv = nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=mid_channels, kernel_size=1, stride=1, padding=0,
                          bias=bias),
                nn.BatchNorm2d(mid_channels),
                activation_layer()
            )
        if stride > 1 and dilation > 1:
            raise ValueError('Can\'t have stride and dilation both > 1 in MobileNetV3')
        self.depthwise_conv = nn.Sequential(
            nn.Conv2d(in_channels=mid_channels, out_channels=mid_channels, kernel_size=kernel_size, stride=stride,
                      padding=(kernel_size - 1) // 2 * dilation, dilation=dilation, groups=mid_channels, bias=bias),
            nn.BatchNorm2d(mid_channels),
            activation_layer()
        )
        if self.with_se:
            self.se = SELayer(channels=mid_channels, ratio=4)

        self.linear_conv = nn.Sequential(
            nn.Conv2d(in_channels=mid_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0,
                      bias=False),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):

        def _inner_forward(x):
            out = x

            if self.with_expand_conv:
                out = self.expand_conv(out)

            out = self.depthwise_conv(out)

            if self.with_se:
                out = self.se(out)

            out = self.linear_conv(out)

            if self.with_res_shortcut:
                return x + out
            else:
                return out

        out = _inner_forward(x)

        return out
