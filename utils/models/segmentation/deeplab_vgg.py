# jcdubron/scnn_pytorch
import torch.nn as nn
from collections import OrderedDict
from ..builder import MODELS


@MODELS.register()
class DeepLabV1(nn.Module):
    def __init__(self, backbone_cfg,
                 num_classes,
                 spatial_conv_cfg=None,
                 lane_classifier_cfg=None,
                 dropout_1=0.1):
        super(DeepLabV1, self).__init__()

        self.encoder = MODELS.from_dict(backbone_cfg)
        self.fc67 = nn.Sequential(
            nn.Conv2d(512, 1024, 3, padding=4, dilation=4, bias=False),
            nn.BatchNorm2d(1024),
            nn.ReLU(),
            nn.Conv2d(1024, 128, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        self.scnn = MODELS.from_dict(spatial_conv_cfg)
        self.fc8 = nn.Sequential(
            nn.Dropout2d(dropout_1),
            nn.Conv2d(128, num_classes, 1)
        )
        self.softmax = nn.Softmax(dim=1)
        self.lane_classifier = MODELS.from_dict(lane_classifier_cfg)

    def forward(self, x):
        out = OrderedDict()

        output = self.encoder(x)
        output = self.fc67(output)

        if self.scnn is not None:
            output = self.scnn(output)

        output = self.fc8(output)
        out['out'] = output
        if self.lane_classifier is not None:
            output = self.softmax(output)
            out['lane'] = self.lane_classifier(output)

        return out
