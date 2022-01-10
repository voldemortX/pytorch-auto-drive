from torch import nn as nn
from torch.nn import functional as F

from ...builder import MODELS
from ..blocks import non_bottleneck_1d


class BilateralUpsamplerBlock(nn.Module):
    # Added a coarse path to the original ERFNet UpsamplerBlock
    # Copied and modified from:
    # https://github.com/ZJULearning/resa/blob/14b0fea6a1ab4f45d8f9f22fb110c1b3e53cf12e/models/decoder.py#L67

    def __init__(self, ninput, noutput):
        super(BilateralUpsamplerBlock, self).__init__()
        self.conv = nn.ConvTranspose2d(ninput, noutput, 3, stride=2, padding=1, output_padding=1, bias=True)
        self.bn = nn.BatchNorm2d(noutput, eps=1e-3, track_running_stats=True)
        self.follows = nn.ModuleList(non_bottleneck_1d(noutput, 0, 1) for _ in range(2))

        # interpolate
        self.interpolate_conv = nn.Conv2d(ninput, noutput, kernel_size=1, bias=False)
        self.interpolate_bn = nn.BatchNorm2d(noutput, eps=1e-3)

    def forward(self, input):
        # Fine branch
        output = self.conv(input)
        output = self.bn(output)
        out = F.relu(output)
        for follow in self.follows:
            out = follow(out)

        # Coarse branch (keep at align_corners=True)
        interpolate_output = self.interpolate_conv(input)
        interpolate_output = self.interpolate_bn(interpolate_output)
        interpolate_output = F.relu(interpolate_output)
        interpolated = F.interpolate(interpolate_output, size=out.shape[-2:], mode='bilinear', align_corners=True)

        return out + interpolated


@MODELS.register()
class BUSD(nn.Module):
    # Bilateral Up-Sampling Decoder in RESA paper,
    # make it work for arbitrary input channels (8x up-sample then predict).
    # Drops transposed prediction layer in ERFNet, while adds an extra up-sampling block.

    def __init__(self, in_channels=128, num_classes=5):
        super(BUSD, self).__init__()
        base = in_channels // 8
        self.layers = nn.ModuleList(BilateralUpsamplerBlock(ninput=base * 2 ** (3 - i), noutput=base * 2 ** (2 - i))
                                    for i in range(3))
        self.output_proj = nn.Conv2d(base, num_classes, kernel_size=1, bias=True)  # Keep bias=True for prediction

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)

        return self.output_proj(x)
