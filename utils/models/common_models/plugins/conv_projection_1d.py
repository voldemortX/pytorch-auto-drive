import torch
import torch.nn as nn
from torch.nn import functional as F

from ...builder import MODELS


@MODELS.register()
class ConvProjection_1D(torch.nn.Module):
    # Projection based on line features (1D convs)
    def __init__(self, num_layers, in_channels, bias=True, k=3):
        # bias is set as True in FCOS
        super().__init__()
        self.num_layers = num_layers
        self.hidden_layers = nn.ModuleList(nn.Conv1d(in_channels if i > 0 else in_channels,
                                                     in_channels, kernel_size=k, bias=bias, padding=(k - 1) // 2)
                                           for i in range(num_layers))
        self.hidden_norms = nn.ModuleList(nn.BatchNorm1d(in_channels) for _ in range(num_layers))

    def forward(self, x):
        for conv, norm in zip(self.hidden_layers, self.hidden_norms):
            x = F.relu(norm(conv(x)))

        return x
