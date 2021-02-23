from collections import OrderedDict

import warnings
from .._utils import IntermediateLayerGetter
from ..utils import load_state_dict_from_url
from .. import resnet
from .deeplab import DeepLabV3Head, DeepLabV2Head, DeepLabV1Head, DeepLab, ReconHead
from .fcn import FCN, FCNHead
from .erfnet import ERFNet
from .deeplab_vgg import DeepLabV1
from .enet import ENet, Encoder
from ..lane_detection import SpatialConv, SimpleLaneExist, RESAReducer
from torch import load
import torch

__all__ = ['fcn_resnet50', 'fcn_resnet101', 'deeplabv2_resnet101', 'deeplabv3_resnet50', 'deeplabv3_resnet101',
           'erfnet_resnet', 'deeplabv1_vgg16', 'enet_',
           'deeplabv1_resnet101', 'deeplabv1_resnet50', 'deeplabv1_resnet34', 'deeplabv1_resnet18']

model_urls = {
    'fcn_resnet50_coco': None,
    'fcn_resnet101_coco': 'https://download.pytorch.org/models/fcn_resnet101_coco-7ecb50ca.pth',
    'deeplabv3_resnet50_coco': None,
    'deeplabv3_resnet101_coco': 'https://download.pytorch.org/models/deeplabv3_resnet101_coco-586e9e4e.pth',
    'deeplabv2_resnet101_coco': None,
    'deeplabv1_resnet18_coco': None,
    'deeplabv1_resnet34_coco': None,
    'deeplabv1_resnet50_coco': None,
    'deeplabv1_resnet101_coco': None
}


def _segm_resnet(name, backbone_name, num_classes, aux, recon_loss, pretrained_backbone=True,
                 num_lanes=0, channel_reduce=0, scnn=False, flattened_size=3965):
    backbone = resnet.__dict__[backbone_name](
        pretrained=pretrained_backbone,
        replace_stride_with_dilation=[False, True, True])

    return_layers = {'layer4': 'out'}
    if aux:
        return_layers['layer3'] = 'aux'
    if recon_loss:
        return_layers['layer2'] = 'recon'
    backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)

    # For lane detection (the style here is fucked up, but lets keep it for now)
    inplanes = 2048 if backbone_name == 'resnet50' or backbone_name == 'resnet101' else 512
    num_channels = inplanes if channel_reduce <= 0 else channel_reduce
    lane_classifier = None
    final_dilation = True  # A dilation of 12 for the final prediction layer as in original DeepLab-LargeFOV
    if num_lanes > 0:
        final_dilation = False  # Lane detection baseline has no dilation in ResNet final prediction
        lane_classifier = SimpleLaneExist(num_output=num_lanes, flattened_size=flattened_size)
    channel_reducer = None
    if channel_reduce > 0:
        if channel_reduce > inplanes:
            raise ValueError
        channel_reducer = RESAReducer(in_channels=inplanes, reduce=channel_reduce)
    scnn_layer = None
    if scnn:
        if channel_reduce != 128:
            warnings.warn('Spatial conv is commonly conducted with 128 channels, not {} channels'.format(
                channel_reduce))
        scnn_layer = SpatialConv(num_channels=num_channels)

    aux_classifier = None
    if aux:
        inplanes = 1024 if backbone_name == 'resnet50' or backbone_name == 'resnet101' else 256
        aux_classifier = FCNHead(inplanes, num_classes)

    recon_classifier = None
    if recon_loss:
        recon_classifier = ReconHead(in_channels=512 if backbone_name == 'resnet50' or backbone_name == 'resnet101'
        else 128)

    model_map = {
        'deeplabv3': (DeepLabV3Head, DeepLab),
        'deeplabv2': (DeepLabV2Head, DeepLab),
        'deeplabv1': (DeepLabV1Head, DeepLab),
        'fcn': (FCNHead, FCN),
    }
    if final_dilation:
        classifier = model_map[name][0](num_channels, num_classes)
    else:
        classifier = model_map[name][0](num_channels, num_classes, 1)
    base_model = model_map[name][1]

    model = base_model(backbone, classifier, aux_classifier, recon_classifier,
                       lane_classifier=lane_classifier, channel_reducer=channel_reducer, scnn_layer=scnn_layer)
    return model


# TODO: Get rid of **kwargs
def _load_model(arch_type, backbone, pretrained, progress, num_classes, aux_loss, recon_loss,
                num_lanes=0, channel_reduce=0, scnn=False, flattened_size=3965, **kwargs):
    if pretrained:
        aux_loss = True
    model = _segm_resnet(arch_type, backbone, num_classes, aux_loss, recon_loss,
                         num_lanes=num_lanes, channel_reduce=channel_reduce, scnn=scnn, flattened_size=flattened_size,
                         **kwargs)
    if pretrained:
        arch = arch_type + '_' + backbone + '_coco'
        model_url = model_urls[arch]
        if model_url is None:
            raise NotImplementedError('pretrained {} is not supported as of now'.format(arch))
        else:
            state_dict = load_state_dict_from_url(model_url, progress=progress)
            model.load_state_dict(state_dict)
    return model


def fcn_resnet50(pretrained=False, progress=True,
                 num_classes=21, aux_loss=None, **kwargs):
    """Constructs a Fully-Convolutional Network model with a ResNet-50 backbone.

    Args:
        pretrained (bool): If True, returns a model pre-trained on COCO train2017 which
            contains the same classes as Pascal VOC
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _load_model('fcn', 'resnet50', pretrained, progress, num_classes, aux_loss, **kwargs)


def fcn_resnet101(pretrained=False, progress=True,
                  num_classes=21, aux_loss=None, recon_loss=False, **kwargs):
    """Constructs a Fully-Convolutional Network model with a ResNet-101 backbone.

    Args:
        pretrained (bool): If True, returns a model pre-trained on COCO train2017 which
            contains the same classes as Pascal VOC
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _load_model('fcn', 'resnet101', pretrained, progress, num_classes, aux_loss, recon_loss, **kwargs)


def deeplabv1_resnet18(pretrained=False, progress=True, num_classes=21, aux_loss=None, recon_loss=False,
                       num_lanes=0, channel_reduce=0, scnn=False, flattened_size=3965, **kwargs):
    """Constructs a DeepLab-LargeFOV model with a ResNet-18 backbone.

    Args:
        pretrained (bool): If True, returns a model pre-trained on COCO train2017 which
            contains the same classes as Pascal VOC
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _load_model('deeplabv1', 'resnet18', pretrained, progress, num_classes, aux_loss, recon_loss,
                       num_lanes=num_lanes, channel_reduce=channel_reduce, scnn=scnn, flattened_size=flattened_size,
                       **kwargs)


def deeplabv1_resnet34(pretrained=False, progress=True, num_classes=21, aux_loss=None, recon_loss=False,
                       num_lanes=0, channel_reduce=0, scnn=False, flattened_size=3965, **kwargs):
    """Constructs a DeepLab-LargeFOV model with a ResNet-34 backbone.

    Args:
        pretrained (bool): If True, returns a model pre-trained on COCO train2017 which
            contains the same classes as Pascal VOC
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _load_model('deeplabv1', 'resnet34', pretrained, progress, num_classes, aux_loss, recon_loss,
                       num_lanes=num_lanes, channel_reduce=channel_reduce, scnn=scnn, flattened_size=flattened_size,
                       **kwargs)


def deeplabv1_resnet50(pretrained=False, progress=True, num_classes=21, aux_loss=None, recon_loss=False,
                       num_lanes=0, channel_reduce=0, scnn=False, flattened_size=3965, **kwargs):
    """Constructs a DeepLab-LargeFOV model with a ResNet-50 backbone.

    Args:
        pretrained (bool): If True, returns a model pre-trained on COCO train2017 which
            contains the same classes as Pascal VOC
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _load_model('deeplabv1', 'resnet50', pretrained, progress, num_classes, aux_loss, recon_loss,
                       num_lanes=num_lanes, channel_reduce=channel_reduce, scnn=scnn, flattened_size=flattened_size,
                       **kwargs)


def deeplabv1_resnet101(pretrained=False, progress=True, num_classes=21, aux_loss=None, recon_loss=False,
                        num_lanes=0, channel_reduce=0, scnn=False, flattened_size=3965, **kwargs):
    """Constructs a DeepLab-LargeFOV model with a ResNet-101 backbone.

    Args:
        pretrained (bool): If True, returns a model pre-trained on COCO train2017 which
            contains the same classes as Pascal VOC
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _load_model('deeplabv1', 'resnet101', pretrained, progress, num_classes, aux_loss, recon_loss,
                       num_lanes=num_lanes, channel_reduce=channel_reduce, scnn=scnn, flattened_size=flattened_size,
                       **kwargs)


def deeplabv2_resnet101(pretrained=False, progress=True,
                        num_classes=21, aux_loss=None, recon_loss=False, **kwargs):
    """Constructs a DeepLabV2 model with a ResNet-101 backbone.

    Args:
        pretrained (bool): If True, returns a model pre-trained on COCO train2017 which
            contains the same classes as Pascal VOC
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _load_model('deeplabv2', 'resnet101', pretrained, progress, num_classes, aux_loss,
                       recon_loss, **kwargs)


def deeplabv3_resnet50(pretrained=False, progress=True,
                       num_classes=21, aux_loss=None, **kwargs):
    """Constructs a DeepLabV3 model with a ResNet-50 backbone.

    Args:
        pretrained (bool): If True, returns a model pre-trained on COCO train2017 which
            contains the same classes as Pascal VOC
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _load_model('deeplabv3', 'resnet50', pretrained, progress, num_classes, aux_loss, **kwargs)


def deeplabv3_resnet101(pretrained=False, progress=True,
                        num_classes=21, aux_loss=None, recon_loss=False, **kwargs):
    """Constructs a DeepLabV3 model with a ResNet-101 backbone.

    Args:
        pretrained (bool): If True, returns a model pre-trained on COCO train2017 which
            contains the same classes as Pascal VOC
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _load_model('deeplabv3', 'resnet101', pretrained, progress, num_classes, aux_loss,
                       recon_loss, **kwargs)


def erfnet_resnet(pretrained_weights='erfnet_encoder_pretrained.pth.tar', num_classes=19, num_lanes=0,
                  dropout_1=0.03, dropout_2=0.3, flattened_size=3965, scnn=False):
    """Constructs a ERFNet model with ResNet-style backbone.

    Args:
        pretrained_weights (str): If not None, load ImageNet pre-trained weights from this filename
    """
    net = ERFNet(num_classes=num_classes, encoder=None, num_lanes=num_lanes, dropout_1=dropout_1, dropout_2=dropout_2,
                 flattened_size=flattened_size, scnn=scnn)
    if pretrained_weights is not None:  # Load ImageNet pre-trained weights
        saved_weights = load(pretrained_weights)['state_dict']
        original_weights = net.state_dict()
        for key in saved_weights.keys():
            my_key = key.replace('module.features.', '')
            if my_key in original_weights.keys():
                original_weights[my_key] = saved_weights[key]
        net.load_state_dict(original_weights)
    return net


def deeplabv1_vgg16(pretrained_weights='pytorch-pretrained', num_classes=19, num_lanes=0,
                    dropout_1=0.1, flattened_size=4500, scnn=False):
    """Constructs a DeepLab-LargeFOV model with a VGG16 backbone, similar to the official DeepLabV1.
       With SCNN modifications (128 channels, max dilation 4).

    Args:
        pretrained_weights (str): If "pytorch-pretrained", load ImageNet pre-trained weights
    """
    pretrain = False
    if pretrained_weights == 'pytorch-pretrained':
        pretrain = True
    net = DeepLabV1(num_classes=num_classes, encoder=None, num_lanes=num_lanes, dropout_1=dropout_1,
                    flattened_size=flattened_size, scnn=scnn, pretrain=pretrain)
    return net


def enet_(num_classes=19, encoder_relu=False, decoder_relu=True, dropout_1=0.01, dropout_2=0.1, num_lanes=0,
          sad=False, flattened_size=4500, encoder_only=False, pretrained_weights=None):
    net = ENet(num_classes=num_classes, encoder_relu=encoder_relu, decoder_relu=decoder_relu, dropout_1=dropout_1,
               dropout_2=dropout_2, num_lanes=num_lanes, sad=sad, flattened_size=flattened_size,
               encoder_only=encoder_only, encoder=None)

    if pretrained_weights is not None:  # Load pre-trained weights
        saved_weights = load(pretrained_weights)['model']
        original_weights = net.state_dict()
        for key in saved_weights.keys():
            if key in original_weights.keys():
                original_weights[key] = saved_weights[key]
        net.load_state_dict(original_weights)

    return net
