# Note that this file only implement DeepLabs with ResNet backbone,
# For the original DeepLab-LargeFOV with VGG backbone, refer to deeplab_vgg.py
import torch
from torch import nn
from torch.nn import functional as F

from ._utils import _SimpleSegmentationModel


__all__ = ["DeepLab"]


class DeepLab(_SimpleSegmentationModel):
    # Should be able to support all deeplab models without CRF
    # Note that v1 is not the official version
    pass


class DeepLabV3Head(nn.Sequential):
    def __init__(self, in_channels, num_classes):
        super(DeepLabV3Head, self).__init__(
            ASPP(in_channels, [12, 24, 36]),
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, num_classes, 1)
        )


# For better format consistency
class DeepLabV2Head(nn.Sequential):
    def __init__(self, in_channels, num_classes):
        super(DeepLabV2Head, self).__init__(
            ASPP_V2(in_channels, num_classes, [6, 12, 18, 24])
        )


# For better format consistency
# Not the official VGG backbone version
class DeepLabV1Head(nn.Sequential):
    def __init__(self, in_channels, num_classes):
        super(DeepLabV1Head, self).__init__(
            LargeFOV(in_channels, num_classes)
        )


class ReconHead(nn.Sequential):
    def __init__(self, in_channels):
        super(ReconHead, self).__init__(
            # Down, Hold, Up (similar to U-Net)
            # 41 -> 81
            nn.Conv2d(in_channels, int(in_channels / 2), 1, bias=False),
            nn.BatchNorm2d(int(in_channels / 2)),
            nn.ReLU(),
            nn.Conv2d(int(in_channels / 2), int(in_channels / 2), 1, bias=False),
            nn.BatchNorm2d(int(in_channels / 2)),
            nn.ReLU(),
            nn.ConvTranspose2d(int(in_channels / 2), int(in_channels / 2), 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(int(in_channels / 2)),
            nn.ReLU(),

            # 81 -> 161
            nn.Conv2d(int(in_channels / 2), int(int(in_channels / 4)), 1, bias=False),
            nn.BatchNorm2d(int(in_channels / 4)),
            nn.ReLU(),
            nn.Conv2d(int(in_channels / 4), int(in_channels / 4), 1, bias=False),
            nn.BatchNorm2d(int(in_channels / 4)),
            nn.ReLU(),
            nn.ConvTranspose2d(int(in_channels / 4), int(in_channels / 4), 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(int(in_channels / 4)),
            nn.ReLU(),

            # 161 -> 321
            nn.Conv2d(int(in_channels / 4), int(int(in_channels / 8)), 1, bias=False),
            nn.BatchNorm2d(int(in_channels / 8)),
            nn.ReLU(),
            nn.Conv2d(int(in_channels / 8), int(in_channels / 8), 1, bias=False),
            nn.BatchNorm2d(int(in_channels / 8)),
            nn.ReLU(),
            nn.ConvTranspose2d(int(in_channels / 8), int(in_channels / 8), 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(int(in_channels / 8)),
            nn.ReLU(),

            # channels -> 3 (is this good enough?)
            nn.Conv2d(int(in_channels / 8), 3, 1, bias=False)

            # nn.Conv2d(int(in_channels / 4), 3, 1, bias=False)
        )


class ASPPConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, dilation):
        modules = [
            nn.Conv2d(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        ]
        super(ASPPConv, self).__init__(*modules)


class ASPPPooling(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(ASPPPooling, self).__init__(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU())

    def forward(self, x):
        size = x.shape[-2:]
        x = super(ASPPPooling, self).forward(x)
        return F.interpolate(x, size=size, mode='bilinear', align_corners=True)


class ASPP(nn.Module):
    def __init__(self, in_channels, atrous_rates):
        super(ASPP, self).__init__()
        out_channels = 256
        modules = []
        modules.append(nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()))

        rate1, rate2, rate3 = tuple(atrous_rates)
        modules.append(ASPPConv(in_channels, out_channels, rate1))
        modules.append(ASPPConv(in_channels, out_channels, rate2))
        modules.append(ASPPConv(in_channels, out_channels, rate3))
        modules.append(ASPPPooling(in_channels, out_channels))

        self.convs = nn.ModuleList(modules)

        self.project = nn.Sequential(
            nn.Conv2d(5 * out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Dropout(0.5))

    def forward(self, x):
        res = []
        for conv in self.convs:
            res.append(conv(x))
        res = torch.cat(res, dim=1)
        return self.project(res)


# Copied and modified from hfslyc/AdvSemiSeg/model
class ASPP_V2(nn.Module):
    def __init__(self, in_channels, num_classes, atrous_rates):
        super(ASPP_V2, self).__init__()
        self.convs = nn.ModuleList()
        for rates in atrous_rates:
            self.convs.append(
                nn.Conv2d(in_channels, num_classes, kernel_size=3, stride=1, padding=rates, dilation=rates, bias=True))

    def forward(self, x):
        res = self.convs[0](x)
        for i in range(len(self.convs) - 1):
            res += self.convs[i + 1](x)
            return res


class LargeFOV(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(LargeFOV, self).__init__()
        self.conv = nn.Conv2d(in_channels, num_classes, kernel_size=3, stride=1, padding=12, dilation=12, bias=True)

    def forward(self, x):
        return self.conv(x)
