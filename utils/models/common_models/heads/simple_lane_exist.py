import torch
from torch import nn as nn
from torch.nn import functional as F

from ...builder import MODELS


@MODELS.register()
class SimpleLaneExist(nn.Module):
    # Typical lane existence head originated from the SCNN paper

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
