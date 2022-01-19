import torch.nn as nn
from torch.nn import functional as F


class PPM(nn.ModuleList):
    """
    Pooling pyramid module used in PSPNet
    Args:
        pool_scales(tuple(int)): Pooling scales used in pooling Pyramid Module
        applied on the last feature. default: (1, 2, 3, 6)
    """

    def __init__(self, pool_scales, in_channels, channels, align_corners=False):
        super(PPM, self).__init__()
        self.pool_scales = pool_scales
        self.align_corners = align_corners
        self.in_channels = in_channels
        self.channels = channels
        for pool_scale in pool_scales:
            self.append(
                nn.Sequential(
                    nn.AdaptiveAvgPool2d(pool_scale),
                    nn.Conv2d(self.in_channels, self.channels, 1),
                    nn.BatchNorm2d(self.channels),
                    nn.ReLU()))

    def forward(self, x):
        ppm_outs = []
        for ppm in self:
            ppm_out = ppm(x)
            upsampled_ppm_out = F.interpolate(ppm_out, size=x.size()[2:], mode='bilinear',
                                              align_corners=self.align_corners)
            ppm_outs.append(upsampled_ppm_out)
        return ppm_outs
