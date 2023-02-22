import math
import torch
from torch import nn as nn
from torch.nn import functional as F

from ...builder import MODELS
from ..._utils import is_tracing


@MODELS.register()
class SpatialConv(nn.Module):
    # SCNN

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

    def build_slice(self, vertical, reverse, shape):
        # refer to https://github.com/harryhan618/SCNN_Pytorch/blob/master/model.py
        slices = []
        if vertical: # h
            concat_dim = 2 
            length = shape[concat_dim]
            for i in range(length):
                slices.append(
                    (slice(None), slice(None), slice(i, i+1), slice(None))
                )
        else: # w
            concat_dim = 3
            length = shape[concat_dim]
            for i in range(length):
                slices.append(
                    (slice(None), slice(None), slice(None), slice(i, i+1))
                )
        if reverse:
            slices = slices[::-1]
        return slices, concat_dim

    def non_inplace_forward(self, input):
        output = input
        vertical = [True, True, False, False]
        reverse = [False, True, False, True]
        convs = [self.conv_d, self.conv_u, self.conv_r, self.conv_l]
        for ver, rev, conv in zip(vertical, reverse, convs):
            slices, dim = self.build_slice(ver, rev, input.shape)
            output_slices = []
            for idx, s in enumerate(slices):
                # the condition is to align with the original forward
                if (rev is False and idx > 0) or (rev is True and idx < len(slices) - 1 and idx > 0):
                    output_slices.append(
                        output[s]  + F.relu(conv(output_slices[-1]))
                    )
                else:
                    output_slices.append(output[s])
            if rev:
                output_slices = output_slices[::-1]
            output = torch.cat(output_slices, dim=dim)
            if ver is False and rev is True:
                break
        return output

    def forward(self, input):
        output = input

        if is_tracing():
            # PyTorch index+add_ will be ignored in traced graph
            # Down
            for i in range(1, output.shape[2]):
                output[:, :, i:i + 1, :] = output[:, :, i:i + 1, :].add(F.relu(self.conv_d(output[:, :, i - 1:i, :])))
            # Up
            for i in range(output.shape[2] - 2, 0, -1):
                output[:, :, i:i + 1, :] = output[:, :, i:i + 1, :].add(
                    F.relu(self.conv_u(output[:, :, i + 1:i + 2, :])))
            # Right
            for i in range(1, output.shape[3]):
                output[:, :, :, i:i + 1] = output[:, :, :, i:i + 1].add(F.relu(self.conv_r(output[:, :, :, i - 1:i])))
            # Left
            for i in range(output.shape[3] - 2, 0, -1):
                output[:, :, :, i:i + 1] = output[:, :, :, i:i + 1].add(
                    F.relu(self.conv_l(output[:, :, :, i + 1:i + 2])))
        else:
            output = self.non_inplace_forward(output)

        return output
