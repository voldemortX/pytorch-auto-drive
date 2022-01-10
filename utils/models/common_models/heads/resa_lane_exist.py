from torch import nn as nn
from torch.nn import functional as F

from ...builder import MODELS


@MODELS.register()
class RESALaneExist(nn.Module):
    def __init__(self, num_output, flattened_size=3965, dropout=0.1, in_channels=128):
        super().__init__()

        self.layers = nn.ModuleList()
        self.layers.append(nn.Dropout2d(dropout))
        self.layers.append(nn.Conv2d(in_channels, num_output + 1, (1, 1), stride=1, padding=(0, 0), bias=True))
        self.pool = nn.AvgPool2d(2, stride=2)
        self.linear1 = nn.Linear(flattened_size, 128)
        self.linear2 = nn.Linear(128, num_output)

    def forward(self, input):
        output = input
        for layer in self.layers:
            output = layer(output)
        output = F.softmax(output, dim=1)
        output = self.pool(output)
        output = output.flatten(start_dim=1)
        output = self.linear1(output)
        output = F.relu(output)
        output = self.linear2(output)

        return output
