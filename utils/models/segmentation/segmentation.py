from ..utils import load_state_dict_from_url
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
