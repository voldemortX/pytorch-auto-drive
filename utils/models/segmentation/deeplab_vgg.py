# Referenced from jcdubron/scnn_pytorch
# DeepLabV1 and variants
import torch.nn as nn
from collections import OrderedDict

from ..builder import MODELS


@MODELS.register()
class DeepLabV1(nn.Module):
    # Original DeepLabV1 with VGG-16 for lane detection
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


@MODELS.register()
class DeepLabV1Lane(nn.Module):
    # General lane baseline
    def __init__(self,
                 backbone_cfg=None,
                 spatial_conv_cfg=None,
                 lane_classifier_cfg=None,
                 reducer_cfg=None,
                 classifier_cfg=None,
                 uper_cfg=None):
        super().__init__()
        self.encoder = MODELS.from_dict(backbone_cfg)
        self.reducer = MODELS.from_dict(reducer_cfg)
        self.scnn = MODELS.from_dict(spatial_conv_cfg)
        self.classifier = MODELS.from_dict(classifier_cfg)
        self.softmax = nn.Softmax(dim=1)
        self.lane_classifier = MODELS.from_dict(lane_classifier_cfg)
        self.uper_decoder = MODELS.from_dict(uper_cfg)

    def forward(self, input):
        out = OrderedDict()
        output = self.encoder(input)
        if self.uper_decoder is not None:
            output = self.uper_decoder(output)
        if self.reducer is not None:
            output = self.reducer(output)
        if self.scnn is not None:
            output = self.scnn(output)
        output = self.classifier(output)
        out['out'] = output
        if self.lane_classifier is not None:
            output = self.softmax(output)
            out['lane'] = self.lane_classifier(output)
        return out


@MODELS.register()
class SegRepVGG(DeepLabV1Lane):
    def eval(self, profiling=False):
        """A secure copy of eval()
        """

        if profiling:
            for module in self.encoder.modules():
                if hasattr(module, 'switch_to_deploy'):
                    module.switch_to_deploy()
            print('Deploy!')
            return self.train(False)
        else:
            return self.train(False)
