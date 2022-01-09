import torch
import torch.nn as nn
from torch.nn import functional as F

from ...builder import MODELS

@MODELS.register()
class FPNHead(nn.Module):
    def __init__(self, in_channels, channels, input_induces=(0, 1, 2, 3), align_corners=False):
        super(FPNHead, self).__init__()
        self.in_channels = in_channels
        self.channels = channels
        self.input_induces = input_induces
        self.align_corners = align_corners
        assert len(self.in_channels) == len(self.in_channels), 'match error'
        # FPN module
        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()
        for in_channel in self.in_channels:
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

    def input_transforms(self, inputs):
        assert isinstance(inputs, tuple), 'input must be a tuple'
        inputs_list = [inputs[i] for i in self.input_induces]
        return inputs_list

    def forward(self, inputs):
        inputs = self.input_transforms(inputs)
        # build laterals
        laterals = [lateral_conv(inputs[i]) for i, lateral_conv in enumerate(self.lateral_convs)]
        used_backbone_levels = len(laterals)
        # top-down path
        for i in range(used_backbone_levels - 1, 0, -1):
            prev_shape = laterals[i - 1].shape[2:]
            laterals[i - 1] = laterals[i - 1] + F.interpolate(laterals[i], size=prev_shape, mode='bilinear',
                                                              align_corners=self.align_corners)
        fpn_outs = [self.fpn_convs[i](laterals[i]) for i in range(used_backbone_levels)]
        for i in range(used_backbone_levels - 1, 0, -1):
            fpn_outs[i] = F.interpolate(fpn_outs[i], fpn_outs[0].shape[2:], mode='bilinear',
                                        align_corners=self.align_corners)
        fpn_outs = torch.cat(fpn_outs, dim=1)
        output = self.fpn_bottleneck(fpn_outs)
        return output


