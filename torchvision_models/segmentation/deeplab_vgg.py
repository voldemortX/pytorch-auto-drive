# jcdubron/scnn_pytorch
import torch.nn as nn
import torchvision
from collections import OrderedDict
from ..lane_detection.common_models import SpatialConv, SimpleLaneExist


# Modified VGG16 backbone in DeepLab-LargeFOV
class VGG16(nn.Module):
    def __init__(self, pretained=True):
        super(VGG16, self).__init__()
        self.pretrained = pretained
        self.net = torchvision.models.vgg16_bn(pretrained=self.pretrained).features
        for i in [34, 37, 40]:
            conv = self.net._modules[str(i)]
            dilated_conv = nn.Conv2d(
                conv.in_channels, conv.out_channels, conv.kernel_size, stride=conv.stride,
                padding=tuple(p * 2 for p in conv.padding), dilation=2, bias=(conv.bias is not None)
            )
            dilated_conv.load_state_dict(conv.state_dict())
            self.net._modules[str(i)] = dilated_conv
        self.net._modules.pop('33')
        self.net._modules.pop('43')

    def forward(self, x):
        x = self.net(x)
        return x


class DeepLabV1(nn.Module):
    def __init__(self, num_classes, encoder=None, num_lanes=0, dropout_1=0.1, flattened_size=3965,
                 scnn=False, pretrain=False):
        super(DeepLabV1, self).__init__()

        if encoder is None:
            self.encoder = VGG16(pretained=pretrain)
        else:
            self.encoder = encoder

        self.fc67 = nn.Sequential(
            nn.Conv2d(512, 1024, 3, padding=4, dilation=4, bias=False),
            nn.BatchNorm2d(1024),
            nn.ReLU(),
            nn.Conv2d(1024, 128, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )

        if scnn:
            self.scnn = SpatialConv()
        else:
            self.scnn = None

        self.fc8 = nn.Sequential(
            nn.Dropout2d(dropout_1),
            nn.Conv2d(128, num_classes, 1)
        )

        self.softmax = nn.Softmax(dim=1)

        if num_lanes > 0:
            self.lane_classifier = SimpleLaneExist(num_output=num_lanes, flattened_size=flattened_size)
        else:
            self.lane_classifier = None

    def forward(self, input):
        out = OrderedDict()

        output = self.encoder(input)
        output = self.fc67(output)

        if self.scnn is not None:
            output = self.scnn(output)

        output = self.fc8(output)
        out['out'] = output
        if self.lane_classifier is not None:
            output = self.softmax(output)
            out['lane'] = self.lane_classifier(output)
        return out

# t = torch.randn(1, 3, 288, 800)
# net = VGG16Net(num_classes=5, encoder=None, aux=5, flattened_size=4500, scnn=True)
# res=net(t)
# print(res['out'].shape)
# print(res['lane'].shape)
