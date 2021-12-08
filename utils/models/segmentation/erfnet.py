# Copied and modified from Eromera/erfnet_pytorch,
# cardwing/Codes-for-Lane-Detection and
# jcdubron/scnn_pytorch

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

from ..common_models import non_bottleneck_1d
from ..builder import MODELS
from ._utils import _EncoderDecoderModel


class DownsamplerBlock(nn.Module):
    def __init__(self, ninput, noutput):
        super().__init__()
        self.conv = nn.Conv2d(ninput, noutput-ninput, (3, 3), stride=2, padding=1, bias=True)
        self.pool = nn.MaxPool2d(2, stride=2)
        self.bn = nn.BatchNorm2d(noutput, eps=1e-3)

    def forward(self, input):
        output = torch.cat([self.conv(input), self.pool(input)], 1)
        output = self.bn(output)
        return F.relu(output)


class Encoder(nn.Module):
    def __init__(self, num_classes, dropout_1=0.03, dropout_2=0.3):
        super().__init__()
        self.initial_block = DownsamplerBlock(3, 16)

        self.layers = nn.ModuleList()

        self.layers.append(DownsamplerBlock(16, 64))

        for x in range(0, 5):  # 5 times
            self.layers.append(non_bottleneck_1d(64, dropout_1, 1))

        self.layers.append(DownsamplerBlock(64, 128))

        for x in range(0, 2):  # 2 times
            self.layers.append(non_bottleneck_1d(128, dropout_2, 2))
            self.layers.append(non_bottleneck_1d(128, dropout_2, 4))
            self.layers.append(non_bottleneck_1d(128, dropout_2, 8))
            self.layers.append(non_bottleneck_1d(128, dropout_2, 16))

        # Only in encoder mode:
        self.output_conv = nn.Conv2d(128, num_classes, 1, stride=1, padding=0, bias=True)

    def forward(self, input, predict=False):
        output = self.initial_block(input)

        for layer in self.layers:
            output = layer(output)

        if predict:
            output = self.output_conv(output)

        return output


class UpsamplerBlock(nn.Module):
    def __init__(self, ninput, noutput):
        super().__init__()
        self.conv = nn.ConvTranspose2d(ninput, noutput, 3, stride=2, padding=1, output_padding=1, bias=True)
        self.bn = nn.BatchNorm2d(noutput, eps=1e-3)

    def forward(self, input):
        output = self.conv(input)
        output = self.bn(output)
        return F.relu(output)


class Decoder(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(UpsamplerBlock(128, 64))
        self.layers.append(non_bottleneck_1d(64, 0, 1))
        self.layers.append(non_bottleneck_1d(64, 0, 1))
        self.layers.append(UpsamplerBlock(64, 16))
        self.layers.append(non_bottleneck_1d(16, 0, 1))
        self.layers.append(non_bottleneck_1d(16, 0, 1))

        self.output_conv = nn.ConvTranspose2d(16, num_classes, 2, stride=2, padding=0, output_padding=0, bias=True)

    def forward(self, input):
        output = input

        for layer in self.layers:
            output = layer(output)

        output = self.output_conv(output)

        return output


# ERFNet
@MODELS.register()
class ERFNet(_EncoderDecoderModel):
    def __init__(self,
                 spatial_conv_cfg,
                 lane_classifier_cfg,
                 num_classes,
                 dropout_1=0.03,
                 dropout_2=0.3,
                 pretrained_weights=None):
        super().__init__()
        self.encoder = Encoder(num_classes=num_classes, dropout_1=dropout_1, dropout_2=dropout_2)
        self.decoder = Decoder(num_classes)
        self.spatial_conv = MODELS.from_dict(spatial_conv_cfg)
        self.lane_classifier = MODELS.from_dict(lane_classifier_cfg)
        self._load_encoder(pretrained_weights)

    def _load_encoder(self, pretrained_weights):
        if pretrained_weights is not None:  # Load ImageNet pre-trained weights
            saved_weights = torch.load(pretrained_weights)['state_dict']
            original_weights = self.state_dict()
            for key in saved_weights.keys():
                my_key = key.replace('module.features.', '')
                if my_key in original_weights.keys():
                    original_weights[my_key] = saved_weights[key]
            self.load_state_dict(original_weights)
        else:
            print('No ImageNet pre-training.')

    def forward(self, x, only_encode=False):
        out = OrderedDict()
        if only_encode:
            return self.encoder.forward(x, predict=True)
        else:
            output = self.encoder(x)    # predict=False by default
            if self.spatial_conv is not None:
                output = self.spatial_conv(output)
            out['out'] = self.decoder.forward(output)

            if self.lane_classifier is not None:
                out['lane'] = self.lane_classifier(output)
            return out
