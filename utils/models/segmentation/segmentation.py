from torch import load
from ..utils import load_state_dict_from_url
from .erfnet import ERFNet
from .enet import ENet
from ._utils import _SimpleSegmentationModel
from ..builder import MODELS

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


def _build_segmentation_net(backbone_cfg, classifier_cfg,
                            aux_classifier_cfg=None, lane_classifier_cfg=None, reducer_cfg=None, spatial_conv_cfg=None):
    if aux_classifier_cfg is not None:  # Fixed for COCO pre-training
        backbone_cfg['return_layer'] = {
            'layer3': 'aux',
            'layer4': 'out'
        }
    backbone = MODELS.from_dict(backbone_cfg)
    classifier = MODELS.from_dict(classifier_cfg)
    aux_classifier = MODELS.from_dict(aux_classifier_cfg)
    lane_classifier = MODELS.from_dict(lane_classifier_cfg)
    channel_reducer = MODELS.from_dict(reducer_cfg)
    scnn_layer = MODELS.from_dict(spatial_conv_cfg)

    model = _SimpleSegmentationModel(
        backbone, classifier, aux_classifier,
        lane_classifier=lane_classifier, channel_reducer=channel_reducer, scnn_layer=scnn_layer)

    return model


@MODELS.register()
def standard_segmentation_model(backbone_cfg, classifier_cfg,
                                progress=True, pretrained=False, arch_type=None, backbone=None,
                                aux_classifier_cfg=None,
                                lane_classifier_cfg=None, reducer_cfg=None, spatial_conv_cfg=None):
    if pretrained:
        assert 'aux_classifier_cfg' is not None
        assert arch_type is not None
        assert backbone is not None
    model = _build_segmentation_net(backbone_cfg, classifier_cfg,
                                    aux_classifier_cfg, lane_classifier_cfg, reducer_cfg, spatial_conv_cfg)
    if pretrained:
        arch = arch_type + '_' + backbone + '_coco'
        model_url = model_urls[arch]
        if model_url is None:
            raise NotImplementedError('COCO pretrained {} is not supported as of now'.format(arch))
        else:
            state_dict = load_state_dict_from_url(model_url, progress=progress)
            model.load_state_dict(state_dict)
    return model


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
