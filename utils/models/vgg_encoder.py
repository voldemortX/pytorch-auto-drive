from torch import nn as nn
from torchvision.models import vgg16_bn

from .builder import MODELS


@MODELS.register()
class VGG16(nn.Module):
    # Modified VGG16 backbone in DeepLab-LargeFOV,
    # note that due to legacy implementation issues,
    # the converted fully-connected layers (FC-6 FC-7) are not included here.
    # jcdubron/scnn_pytorch

    def __init__(self, pretrained=True):
        super().__init__()
        self.pretrained = pretrained
        self.net = vgg16_bn(pretrained=self.pretrained).features
        for i in [34, 37, 40]:
            conv = self.net._modules[str(i)]
            dilated_conv = nn.Conv2d(
                conv.in_channels, conv.out_channels, conv.kernel_size, stride=conv.stride,
                padding=tuple(p * 2 for p in conv.padding), dilation=2, bias=(conv.bias is not None)
            )
            dilated_conv.load_state_dict(conv.state_dict())
            self.net._modules[str(i)] = dilated_conv
        self.net._modules.pop('33')
        self.net._modules.pop('43')

    def forward(self, x):
        x = self.net(x)

        return x
