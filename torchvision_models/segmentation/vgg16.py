# jcdubron/scnn_pytorch
import math
import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F
from collections import OrderedDict


# Really tricky without global pooling
class LaneExistVgg(nn.Module):
    def __init__(self, num_output, flattened_size=4500):
        super().__init__()
        self.avgpool = nn.AvgPool2d(2, 2)
        self.linear1 = nn.Linear(flattened_size, 128)
        self.linear2 = nn.Linear(128, num_output)

    def forward(self, input, predict=False):
        # print(output.shape)
        output = self.avgpool(input)
        output = output.flatten(start_dim=1)
        output = self.linear1(output)
        output = F.relu(output)
        output = self.linear2(output)
        if predict:
            output = torch.sigmoid(output)
        return output


# SCNN head
class SpatialConv(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_d = nn.Conv2d(128, 128, (1, 9), padding=(0, 4))
        self.conv_u = nn.Conv2d(128, 128, (1, 9), padding=(0, 4))
        self.conv_r = nn.Conv2d(128, 128, (9, 1), padding=(4, 0))
        self.conv_l = nn.Conv2d(128, 128, (9, 1), padding=(4, 0))
        self._adjust_initializations()

    def _adjust_initializations(self) -> None:
        # https://github.com/XingangPan/SCNN/issues/82
        bound = math.sqrt(2.0 / (128 * 9 * 5))
        nn.init.uniform_(self.conv_d.weight, -bound, bound)
        nn.init.uniform_(self.conv_u.weight, -bound, bound)
        nn.init.uniform_(self.conv_r.weight, -bound, bound)
        nn.init.uniform_(self.conv_l.weight, -bound, bound)

    def forward(self, input):
        output = input

        # First one remains unchanged (according to the original paper), why not add a relu afterwards?
        # Update and send to next
        # Down
        for i in range(1, output.shape[2]):
            output[:, :, i:i+1, :].add_(F.relu(self.conv_d(output[:, :, i-1:i, :])))
        # Up
        for i in range(output.shape[2] - 2, 0, -1):
            output[:, :, i:i+1, :].add_(F.relu(self.conv_u(output[:, :, i+1:i+2, :])))
        # Right
        for i in range(1, output.shape[3]):
            output[:, :, :, i:i+1].add_(F.relu(self.conv_r(output[:, :, :, i-1:i])))
        # Left
        for i in range(output.shape[3] - 2, 0, -1):
            output[:, :, :, i:i+1].add_(F.relu(self.conv_l(output[:, :, :, i+1:i+2])))

        return output


class Vgg16(nn.Module):
    def __init__(self, pretained=True):
        super(Vgg16, self).__init__()
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


class VGG16Net(nn.Module):
    def __init__(self, num_classes, encoder=None, aux=0, dropout_1=0.1, dropout_2=0.3, flattened_size=3965,
                 scnn=False, pretrain=False):
        super(VGG16Net, self).__init__()

        if encoder is None:
            self.encoder = Vgg16(pretained=pretrain)
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
            nn.Conv2d(128, 5, 1)
        )

        self.act = nn.Sequential(
            nn.Softmax(dim=1)
        )

        if aux > 0:
            self.aux_head = LaneExistVgg(num_output=aux, flattened_size=flattened_size)
        else:
            self.aux_head = None

    def forward(self, input, only_encode=False):
        out = OrderedDict()

        output = self.encoder(input)
        output = self.fc67(output)

        if self.scnn is not None:
            output = self.scnn(output)

        output = self.fc8(output)
        out['out'] = output
        output = self.act(output)
        if self.aux_head is not None:
            out['aux'] = self.aux_head(output)

        return out

# t = torch.randn(1, 3, 288, 800)
# net = VGG16Net(num_classes=5, encoder=None, aux=5, flattened_size=4500, scnn=True)
# res=net(t)
# print(res['out'].shape)
# print(res['aux'].shape)
