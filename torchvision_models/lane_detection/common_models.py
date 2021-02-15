import math
import torch
import torch.nn as nn
from torch.nn import functional as F


# Unused
# SCNN original decoder for ResNet-101, very large channels, maybe impossible to add anything
# resnet-101 -> H x W x 2048
# 3x3 Conv -> H x W x 512
# Dropout 0.1
# 1x1 Conv -> H x W x 5
# https://github.com/XingangPan/SCNN/issues/35
class SCNNDecoder(nn.Module):
    def __init__(self, in_channels=2048, num_classes=5):
        super(SCNNDecoder, self).__init__()
        out_channels = in_channels // 4
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.dropout1 = nn.Dropout2d(0.1)
        self.conv2 = nn.Conv2d(out_channels, num_classes, 1, bias=False)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout1(x)
        x = self.conv2(x)

        return x


# Reduce channel (typically to 128)
class RESAReducer(nn.Module):
    def __init__(self, in_channels=512, reduce=128):
        super(RESAReducer, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, reduce, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(reduce)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)

        return x


# SCNN head
class SpatialConv(nn.Module):
    def __init__(self, num_channels=128):
        super().__init__()
        self.conv_d = nn.Conv2d(num_channels, num_channels, (1, 9), padding=(0, 4))
        self.conv_u = nn.Conv2d(num_channels, num_channels, (1, 9), padding=(0, 4))
        self.conv_r = nn.Conv2d(num_channels, num_channels, (9, 1), padding=(4, 0))
        self.conv_l = nn.Conv2d(num_channels, num_channels, (9, 1), padding=(4, 0))
        self._adjust_initializations(num_channels=num_channels)

    def _adjust_initializations(self, num_channels=128):
        # https://github.com/XingangPan/SCNN/issues/82
        bound = math.sqrt(2.0 / (num_channels * 9 * 5))
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


# Typical lane existence head originated from the SCNN paper
class SimpleLaneExist(nn.Module):
    def __init__(self, num_output, flattened_size=4500):
        super().__init__()
        self.avgpool = nn.AvgPool2d(2, 2)
        self.linear1 = nn.Linear(flattened_size, 128)
        self.linear2 = nn.Linear(128, num_output)

    def forward(self, input, predict=False):
        # input: logits
        output = self.avgpool(input)
        output = output.flatten(start_dim=1)
        output = self.linear1(output)
        output = F.relu(output)
        output = self.linear2(output)
        if predict:
            output = torch.sigmoid(output)

        return output
