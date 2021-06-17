# TODO: Refactor to a directory
import math
import torch
import torch.nn as nn
from torch.nn import functional as F
from segmentation.erfnet import non_bottleneck_1d


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


# Added a coarse path to the original ERFNet UpsamplerBlock
# Copied and modified from:
# https://github.com/ZJULearning/resa/blob/14b0fea6a1ab4f45d8f9f22fb110c1b3e53cf12e/models/decoder.py#L67
class BilateralUpsamplerBlock(nn.Module):
    def __init__(self, ninput, noutput):
        super(BilateralUpsamplerBlock, self).__init__()
        self.conv = nn.ConvTranspose2d(ninput, noutput, 3, stride=2, padding=1, output_padding=1, bias=True)
        self.bn = nn.BatchNorm2d(noutput, eps=1e-3, track_running_stats=True)
        self.follows = nn.ModuleList(non_bottleneck_1d(noutput, 0, 1) for _ in range(2))

        # interpolate
        self.interpolate_conv = nn.Conv2d(ninput, noutput, kernel_size=1, bias=False)
        self.interpolate_bn = nn.BatchNorm2d(noutput, eps=1e-3)

    def forward(self, input):
        # Fine branch
        output = self.conv(input)
        output = self.bn(output)
        out = F.relu(output)
        for follow in self.follows:
            out = follow(out)

        # Coarse branch (keep at align_corners=True)
        interpolate_output = self.interpolate_conv(input)
        interpolate_output = self.interpolate_bn(interpolate_output)
        interpolate_output = F.relu(interpolate_output)
        interpolated = F.interpolate(interpolate_output, size=out.shape[-2:], mode='bilinear', align_corners=True)

        return out + interpolated


# Bilateral Up-Sampling Decoder in RESA paper,
# make it work for arbitrary input channels (8x up-sample then predict).
# Drops transposed prediction layer in ERFNet, while adds an extra up-sampling block.
class BUSD(nn.Module):
    def __init__(self, in_channels=128, num_classes=5):
        super(BUSD, self).__init__()
        base = in_channels / 8
        self.layers = nn.ModuleList(BilateralUpsamplerBlock(ninput=base * 2 ** (3 - i), noutput=base * 2 ** (2 - i))
                                    for i in range(3))
        self.output_proj = nn.Conv2d(base, num_classes, kernel_size=1, bias=True)  # Keep bias=True for prediction

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)

        return self.output_proj(x)


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
            output[:, :, i:i + 1, :].add_(F.relu(self.conv_d(output[:, :, i - 1:i, :])))
        # Up
        for i in range(output.shape[2] - 2, 0, -1):
            output[:, :, i:i + 1, :].add_(F.relu(self.conv_u(output[:, :, i + 1:i + 2, :])))
        # Right
        for i in range(1, output.shape[3]):
            output[:, :, :, i:i + 1].add_(F.relu(self.conv_r(output[:, :, :, i - 1:i])))
        # Left
        for i in range(output.shape[3] - 2, 0, -1):
            output[:, :, :, i:i + 1].add_(F.relu(self.conv_l(output[:, :, :, i + 1:i + 2])))

        return output


# REcurrent Feature-Shift Aggregator in RESA paper
class RESA(nn.Module):
    def __init__(self, num_channels=128, iteration=5, alpha=2.0):
        super(RESA, self).__init__()
        # Different from SCNN, RESA uses bias=False & different convolution layers for each stride,
        # i.e. 4 * iteration layers vs. 4 layers in SCNN, maybe special init is not needed anymore:
        # https://github.com/ZJULearning/resa/blob/14b0fea6a1ab4f45d8f9f22fb110c1b3e53cf12e/models/resa.py#L21
        self.iteration = iteration
        self.alpha = alpha
        self.conv_d = nn.ModuleList(nn.Conv2d(num_channels, num_channels, (1, 9), padding=(0, 4), bias=False)
                                    for _ in range(iteration))
        self.conv_u = nn.ModuleList(nn.Conv2d(num_channels, num_channels, (1, 9), padding=(0, 4), bias=False)
                                    for _ in range(iteration))
        self.conv_r = nn.ModuleList(nn.Conv2d(num_channels, num_channels, (9, 1), padding=(4, 0), bias=False)
                                    for _ in range(iteration))
        self.conv_l = nn.ModuleList(nn.Conv2d(num_channels, num_channels, (9, 1), padding=(4, 0), bias=False)
                                    for _ in range(iteration))

    def forward(self, x):
        y = x
        h, w = y.shape[-2:]
        if 2 ** self.iteration > max(h, w):
            print('Too many iterations for RESA, your image size may be too small.')

        # We do indexing here to avoid extra input parameters at __init__(), with almost none computation overhead.
        # Also, now it won't block arbitrary shaped input.
        # Down
        for i in range(self.iteration):
            idx = (torch.arange(h) + h // 2 ** (self.iteration - i)) % h
            y.add_(self.alpha * F.relu(self.conv_d[i](y[:, :, idx, :])))
        # Up
        for i in range(self.iteration):
            idx = (torch.arange(h) - h // 2 ** (self.iteration - i)) % h
            y.add_(self.alpha * F.relu(self.conv_u[i](y[:, :, idx, :])))
        # Right
        for i in range(self.iteration):
            idx = (torch.arange(w) + w // 2 ** (self.iteration - i)) % w
            y.add_(self.alpha * F.relu(self.conv_r[i](y[:, :, :, idx])))
        # Left
        for i in range(self.iteration):
            idx = (torch.arange(w) - w // 2 ** (self.iteration - i)) % w
            y.add_(self.alpha * F.relu(self.conv_l[i](y[:, :, :, idx])))

        return y


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


# Lane exist head for ERFNet and ENet
# Really tricky without global pooling
class EDLaneExist(nn.Module):
    def __init__(self, num_output, flattened_size=3965, dropout=0.1, pool='avg'):
        super().__init__()

        self.layers = nn.ModuleList()
        self.layers.append(nn.Conv2d(128, 32, (3, 3), stride=1, padding=(4, 4), bias=False, dilation=(4, 4)))
        self.layers.append(nn.BatchNorm2d(32, eps=1e-03))

        self.layers_final = nn.ModuleList()
        self.layers_final.append(nn.Dropout2d(dropout))
        self.layers_final.append(nn.Conv2d(32, 5, (1, 1), stride=1, padding=(0, 0), bias=True))

        if pool == 'max':
            self.pool = nn.MaxPool2d(2, stride=2)
        elif pool == 'avg':
            self.pool = nn.AvgPool2d(2, stride=2)
        else:
            raise RuntimeError("This type of pool has not been defined yet!")

        self.linear1 = nn.Linear(flattened_size, 128)
        self.linear2 = nn.Linear(128, num_output)

    def forward(self, input):
        output = input
        for layer in self.layers:
            output = layer(output)

        output = F.relu(output)

        for layer in self.layers_final:
            output = layer(output)

        output = F.softmax(output, dim=1)
        output = self.pool(output)
        output = output.flatten(start_dim=1)
        output = self.linear1(output)
        output = F.relu(output)
        output = self.linear2(output)

        return output
