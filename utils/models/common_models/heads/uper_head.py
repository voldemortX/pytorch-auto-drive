import torch
import torch.nn as nn
from torch.nn import functional as F

from ...builder import MODELS
from ..plugins.ppm import PPM


@MODELS.register()
class UperHead(nn.Module):
    def __init__(self, in_channels, channels, pool_scales=(1, 2, 3, 6), align_corners=False):
        super(UperHead, self).__init__()
        self.in_channels = in_channels
        self.channels = channels
        self.align_corners = align_corners
        # PSP module
        self.psp_modules = PPM(pool_scales=pool_scales, in_channels=self.in_channels[-1], channels=self.channels,
                               align_corners=align_corners)
        self.psp_bottleneck = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels[-1] + len(pool_scales) * self.channels, out_channels=self.channels,
                      kernel_size=3, padding=1),
            nn.BatchNorm2d(self.channels),
            nn.ReLU()
        )
        # FPN module
        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()
        for in_channel in self.in_channels[:-1]:
            lateral_conv = nn.Sequential(
                nn.Conv2d(in_channel, self.channels, 1),
                nn.BatchNorm2d(self.channels),
                nn.ReLU())
            fpn_conv = nn.Sequential(
                nn.Conv2d(self.channels, self.channels, 3, padding=1),
                nn.BatchNorm2d(self.channels),
                nn.ReLU()
            )
            self.lateral_convs.append(lateral_conv)
            self.fpn_convs.append(fpn_conv)
        self.fpn_bottleneck = nn.Sequential(
            nn.Conv2d(len(self.in_channels) * self.channels, self.channels, 3, padding=1),
            nn.BatchNorm2d(self.channels),
            nn.ReLU()
        )

    def psp_forward(self, inputs):
        # forward function for psp module
        x = inputs[-1]
        psp_outs = [x]
        psp_outs.extend(self.psp_modules(x))
        psp_outs = torch.cat(psp_outs, dim=1)
        outputs = self.psp_bottleneck(psp_outs)

        return outputs

    def forward(self, inputs):
        assert isinstance(inputs, tuple), 'inputs must be a tuple'
        inputs = list(inputs)
        # build laterals
        laterals = [lateral_conv(inputs[i]) for i, lateral_conv in enumerate(self.lateral_convs)]
        laterals.append(self.psp_forward(inputs))
        # build top-down path
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            prev_shape = laterals[i - 1].shape[2:]
            laterals[i - 1] = laterals[i - 1] + F.interpolate(laterals[i], size=prev_shape, mode='bilinear',
                                                              align_corners=self.align_corners)
        # build outputs
        fpn_outs = [self.fpn_convs[i](laterals[i]) for i in range(used_backbone_levels - 1)]
        # add psp feature
        fpn_outs.append(laterals[-1])
        for i in range(used_backbone_levels - 1, 0, -1):
            fpn_outs[i] = F.interpolate(fpn_outs[i], fpn_outs[0].shape[2:], mode='bilinear',
                                       align_corners=self.align_corners)
        fpn_outs = torch.cat(fpn_outs, dim=1)
        output = self.fpn_bottleneck(fpn_outs)
        return output
