# Adapted from ZJULearning/resa
# Better to use a decoupled implementation,
# costs more codes, but clear.
# Diff from RESA official code:
# 1. we use BN+ReLU in channel reducer
# 2. we always use the BUSD decoder in the paper (official code does not use BUSD in CULane)
# 3. we always use 5 RESA iterations (4 in official code)
# 4. we use a higher capacity lane existence classifier (same as ERFNet/ENet baseline)
# 5. we use the SCNN sqrt(5) init trick for RESA, which
# 5.1. enables fewer warmup steps
# 5.2. combined with 4, produces slightly better performance
# 6. we do not use horizontal flip or cutting height in loading, in which
# 6.1. flip does not help performance (at least on the val set)
# 6.2. w.o. cutting height trick probably is the main reason for our lower performance, but we can't use it since
# other pytorch-auto-drive models do not use it.
import torch.nn as nn
from ..common_models import RESA, RESAReducer, BUSD, RESALaneExist, EDLaneExist, PlainDecoder
from .._utils import IntermediateLayerGetter
from .. import resnet
from ..segmentation.deeplab_vgg import VGG16


class RESANet(nn.Module):
    def __init__(self, num_classes, backbone_name, flattened_size, channel_reduce, pretrained_backbone=True):
        super(RESANet, self).__init__()
        if backbone_name == 'vgg16':
            # VGG16 with dilation
            vgg = VGG16(pretained=True)
            fc6 = nn.Sequential(
                nn.Conv2d(512, 1024, 3, padding=4, dilation=4, bias=False),
                nn.BatchNorm2d(1024),
                nn.ReLU(),
                # nn.Conv2d(1024, 128, 1, bias=False),
                # nn.BatchNorm2d(128),
                # nn.ReLU()
            )
            self.backbone = nn.Sequential(
                vgg,
                fc6
            )
            in_channels = 1024
        else:
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
        self.lane_classifier = EDLaneExist(num_output=num_classes - 1, flattened_size=flattened_size)
        # self.lane_classifier = RESALaneExist(num_output=num_classes - 1, flattened_size=flattened_size)

    def forward(self, x):
        x = self.backbone(x)['out']
        x = self.channel_reducer(x)
        x = self.spatial_conv(x)

        res = {'out': self.decoder(x),
               'lane': self.lane_classifier(x)}

        return res
