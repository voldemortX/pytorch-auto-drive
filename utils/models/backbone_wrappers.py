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


