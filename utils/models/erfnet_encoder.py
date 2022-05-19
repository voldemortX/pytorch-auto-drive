# modified from utils/models/segmentation/erfnet.py
# load pretrained weights during initialization of encoder

import torch
import torch.nn as nn
import torch.nn.functional as F

from .common_models import non_bottleneck_1d
from .builder import MODELS


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

@MODELS.register()
class ERFNetEncoder(nn.Module):
    def __init__(self, num_classes, dropout_1=0.03, dropout_2=0.3, pretrained_weights=None):
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

        # need to initialize the weights
        if pretrained_weights is not None:
            self._load_encoder_weights(pretrained_weights) # Load ImageNet pre-trained weights
        else:
            self._init_weights() # initialize random weights

    def _init_weights(self):
        pass

    def _load_encoder_weights(self, pretrained_weights):
        # load weights from given file path
        try:
            saved_weights = torch.load(pretrained_weights)['state_dict']
        except FileNotFoundError:
            raise FileNotFoundError('pretrained_weights is not there! '
                                    'Please set pretrained_weights=None if you are only testing.')
        original_weights = self.state_dict()
        for key in saved_weights.keys():
            my_key = key.replace('module.features.', '')
            if my_key in original_weights.keys():
                original_weights[my_key] = saved_weights[key]
        self.load_state_dict(original_weights)

    def forward(self, input):
        output = self.initial_block(input)
        for layer in self.layers:
            output = layer(output)

        return output


