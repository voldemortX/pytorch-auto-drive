import torch.nn as nn
from torchvision.models import vgg16_bn

from . import resnet
from .resnet import _resnet, Bottleneck, BasicBlock
from .builder import MODELS
from ._utils import IntermediateLayerGetter


# Maybe it is safe to assume no new block types
block_map = {
    'BasicBlock': BasicBlock,
    'BottleNeck': Bottleneck
}


def parse_return_layers(return_layer):
    if isinstance(return_layer, str):
        return {return_layer: 'out'}
    elif isinstance(return_layer, dict):
        return return_layer
    else:
        raise TypeError('return_layer can either be direct dict or a string, not {}'.format(type(return_layer)))


@MODELS.register()
def free_resnet_backbone(arch, block, layers, pretrained, return_layer, **kwargs):
    block = block_map[block]
    net = _resnet(arch, block, layers, pretrained, progress=True, **kwargs)
    return_layers = parse_return_layers(return_layer)

    return IntermediateLayerGetter(net, return_layers=return_layers)


@MODELS.register()
def predefined_resnet_backbone(backbone_name, return_layer, **kwargs):
    backbone = resnet.__dict__[backbone_name](**kwargs)
    return_layers = parse_return_layers(return_layer)

    return IntermediateLayerGetter(backbone, return_layers=return_layers)


# Modified VGG16 backbone in DeepLab-LargeFOV
# jcdubron/scnn_pytorch
@MODELS.register()
class VGG16(nn.Module):
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
