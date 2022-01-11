from torch import nn as nn
from torch.nn import functional as F

from ...builder import MODELS


@MODELS.register()
class EDLaneExist(nn.Module):
    # Lane exist head for ERFNet, ENet
    # Really tricky without global pooling

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
