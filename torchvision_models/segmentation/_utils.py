import math
import torch
from torch import nn
from torch.nn import functional as F
from collections import OrderedDict


class _SimpleSegmentationModel(nn.Module):
    def __init__(self, backbone, classifier, aux_classifier=None, recon_head=None):
        super(_SimpleSegmentationModel, self).__init__()
        self.backbone = backbone
        self.classifier = classifier
        self.aux_classifier = aux_classifier
        self.recon_head = recon_head

    def forward(self, x):
        # input_shape = x.shape[-2:]
        # contract: features is a dict of tensors
        features = self.backbone(x)

        result = OrderedDict()
        x = features["out"]
        x = self.classifier(x)
        # x = F.interpolate(x, size=(513, 513), mode='bilinear', align_corners=True)
        result["out"] = x

        if self.aux_classifier is not None:
            x = features["aux"]
            x = self.aux_classifier(x)
            # x = F.interpolate(x, size=input_shape, mode='bilinear', align_corners=True)
            result["aux"] = x

        # Reconstruction
        if self.recon_head is not None:
            x = features["recon"]
            x = self.recon_head(x)
            result["recon"] = x

        return result


# SCNN head
class _SpatialConv(nn.Module):
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


# Typical lane existence head originated from the SCNN paper
class _SimpleLaneExist(nn.Module):
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
