import math
import torch
from torch import nn as nn
from torch.nn import functional as F

from ...builder import MODELS
from ..._utils import is_tracing


@MODELS.register()
class RESA(nn.Module):
    # REcurrent Feature-Shift Aggregator in RESA paper

    def __init__(self, num_channels=128, iteration=5, alpha=2.0, trace_arg=None, os=8):
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
        self._adjust_initializations(num_channels=num_channels)
        if trace_arg is not None:  # Pre-compute offsets for a TensorRT supported implementation
            h = (trace_arg['h'] - 1) // os + 1
            w = (trace_arg['w'] - 1) // os + 1
            self.offset_h = []
            self.offset_w = []
            for i in range(self.iteration):
                self.offset_h.append(h // 2 ** (self.iteration - i))
                self.offset_w.append(w // 2 ** (self.iteration - i))

    def _adjust_initializations(self, num_channels=128):
        # https://github.com/XingangPan/SCNN/issues/82
        bound = math.sqrt(2.0 / (num_channels * 9 * 5))
        for i in self.conv_d:
            nn.init.uniform_(i.weight, -bound, bound)
        for i in self.conv_u:
            nn.init.uniform_(i.weight, -bound, bound)
        for i in self.conv_r:
            nn.init.uniform_(i.weight, -bound, bound)
        for i in self.conv_l:
            nn.init.uniform_(i.weight, -bound, bound)

    def forward(self, x):
        y = x
        h, w = y.shape[-2:]
        if 2 ** self.iteration > min(h, w):
            print('Too many iterations for RESA, your image size may be too small.')

        # We do indexing here to avoid extra input parameters at __init__(), with almost none computation overhead.
        # Also, now it won't block arbitrary shaped input.
        # However, we still need an alternative to Gather for TensorRT
        # Down
        for i in range(self.iteration):
            if is_tracing():
                temp = torch.cat([y[:, :, self.offset_h[i]:, :], y[:, :, :self.offset_h[i], :]], dim=-2)
                y = y.add(self.alpha * F.relu(self.conv_d[i](temp)))
            else:
                idx = (torch.arange(h) + h // 2 ** (self.iteration - i)) % h
                y.add_(self.alpha * F.relu(self.conv_d[i](y[:, :, idx, :])))
        # Up
        for i in range(self.iteration):
            if is_tracing():
                temp = torch.cat([y[:, :, (h - self.offset_h[i]):, :], y[:, :, :(h - self.offset_h[i]), :]], dim=-2)
                y = y.add(self.alpha * F.relu(self.conv_u[i](temp)))
            else:
                idx = (torch.arange(h) - h // 2 ** (self.iteration - i)) % h
                y.add_(self.alpha * F.relu(self.conv_u[i](y[:, :, idx, :])))
        # Right
        for i in range(self.iteration):
            if is_tracing():
                temp = torch.cat([y[:, :, :, self.offset_w[i]:], y[:, :, :, :self.offset_w[i]]], dim=-1)
                y = y.add(self.alpha * F.relu(self.conv_r[i](temp)))
            else:
                idx = (torch.arange(w) + w // 2 ** (self.iteration - i)) % w
                y.add_(self.alpha * F.relu(self.conv_r[i](y[:, :, :, idx])))
        # Left
        for i in range(self.iteration):
            if is_tracing():
                temp = torch.cat([y[:, :, :, (w - self.offset_w[i]):], y[:, :, :, :(w - self.offset_w[i])]], dim=-1)
                y = y.add(self.alpha * F.relu(self.conv_l[i](temp)))
            else:
                idx = (torch.arange(w) - w // 2 ** (self.iteration - i)) % w
                y.add_(self.alpha * F.relu(self.conv_l[i](y[:, :, :, idx])))

        return y
