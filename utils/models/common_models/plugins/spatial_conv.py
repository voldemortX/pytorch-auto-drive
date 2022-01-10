import math
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
