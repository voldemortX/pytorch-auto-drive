# Adapted from ZJULearning/resa
# Better to use a decoupled implementation,
# costs more codes, but clear
import torch.nn as nn
from ..common_models import RESA, RESAReducer, BUSD, RESALaneExist, PlainDecoder
from .._utils import IntermediateLayerGetter
from .. import resnet


class RESANet(nn.Module):
    def __init__(self, num_classes, backbone_name, flattened_size, channel_reduce, pretrained_backbone=True):
        super(RESANet, self).__init__()
        backbone = resnet.__dict__[backbone_name](
            pretrained=pretrained_backbone,
            replace_stride_with_dilation=[False, True, True])
        return_layers = {'layer3': 'out'}
        self.backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)
        in_channels = 1024 if backbone_name == 'resnet50' or backbone_name == 'resnet101' else 256
        # self.channel_reducer = RESAReducer(in_channels=in_channels, reduce=channel_reduce, bn_relu=False)
        self.channel_reducer = RESAReducer(in_channels=in_channels, reduce=channel_reduce)
        self.spatial_conv = RESA()
        self.decoder = BUSD(num_classes=num_classes)
        # self.decoder = PlainDecoder(num_classes=num_classes)
        # self.lane_classifier = EDLaneExist(num_output=num_classes - 1, flattened_size=flattened_size)
        self.lane_classifier = RESALaneExist(num_output=num_classes - 1, flattened_size=flattened_size)

    def forward(self, x):
        x = self.backbone(x)['out']
        x = self.channel_reducer(x)
        x = self.spatial_conv(x)

        res = {'out': self.decoder(x),
               'lane': self.lane_classifier(x)}

        return res
