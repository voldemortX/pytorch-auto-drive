import torch
import torch.nn as nn
from mmcv.ops import ModulatedDeformConv2d, modulated_deform_conv2d  # DCNv2
from torch.nn import functional as F
from ...builder import MODELS


@MODELS.register()
class DCN_v2_Ref(ModulatedDeformConv2d):
    """A Encapsulation that acts as normal Conv
    layers. Modified from mmcv's DCNv2.

    Args:
        in_channels (int): Same as nn.Conv2d.
        out_channels (int): Same as nn.Conv2d.
        kernel_size (int or tuple[int]): Same as nn.Conv2d.
        stride (int): Same as nn.Conv2d, while tuple is not supported.
        padding (int): Same as nn.Conv2d, while tuple is not supported.
        dilation (int): Same as nn.Conv2d, while tuple is not supported.
        groups (int): Same as nn.Conv2d.
        bias (bool or str): If specified as `auto`, it will be decided by the
            norm_cfg. Bias will be set as True if norm_cfg is None, otherwise
            False.
    """

    _version = 2

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.conv_offset = nn.Conv2d(
            self.in_channels * 2,
            self.deform_groups * 3 * self.kernel_size[0] * self.kernel_size[1],
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            bias=True)
        self.init_weights()

    def init_weights(self):
        super().init_weights()
        if hasattr(self, 'conv_offset'):
            self.conv_offset.weight.data.zero_()
            self.conv_offset.bias.data.zero_()

    def forward(self, x, ref):
        concat = torch.cat([x, ref], dim=1)
        out = self.conv_offset(concat)
        o1, o2, mask = torch.chunk(out, 3, dim=1)
        offset = torch.cat((o1, o2), dim=1)
        mask = torch.sigmoid(mask)

        return modulated_deform_conv2d(x, offset, mask, self.weight, self.bias,
                                       self.stride, self.padding,
                                       self.dilation, self.groups,
                                       self.deform_groups)


@MODELS.register()
class FeatureFlipFusion(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.proj1 = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, padding=0),
            nn.BatchNorm2d(channels)
        )
        self.proj2_conv = DCN_v2_Ref(channels, channels, kernel_size=(3, 3), padding=1)
        self.proj2_norm = nn.BatchNorm2d(channels)

    def forward(self, feature):
        # B x C x H x W
        flipped = feature.flip(-1)  # An auto-copy

        feature = self.proj1(feature)
        flipped = self.proj2_conv(flipped, feature)
        flipped = self.proj2_norm(flipped)

        return F.relu(feature + flipped)
