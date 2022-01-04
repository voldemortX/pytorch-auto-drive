# Modified from mmsegmentation code, referenced from torchvision

import torch.nn as nn
from ..builder import MODELS
from .._utils import make_divisible
from ..common_models import InvertedResidual
from ..utils import load_state_dict_from_url


@MODELS.register()
class MobileNetV2Encoder(nn.Module):
    """MobileNetV2 backbone (up to second-to-last feature map).
    This backbone is the implementation of
    `MobileNetV2: Inverted Residuals and Linear Bottlenecks
    <https://arxiv.org/abs/1801.04381>`_.
    Args:
        widen_factor (float): Width multiplier, multiply number of
            channels in each layer by this amount. Default: 1.0.
        strides (Sequence[int], optional): Strides of the first block of each
            layer. If not specified, default config in ``arch_setting`` will
            be used.
        dilations (Sequence[int]): Dilation of each layer.
        out_indices (None or Sequence[int]): Output from which stages.
            Default: (7, ).
        frozen_stages (int): Stages to be frozen (all param fixed).
            Default: -1, which means not freezing any parameters.
        norm_eval (bool): Whether to set norm layers to eval mode, namely,
            freeze running stats (mean and var). Note: Effect on Batch Norm
            and its variants only. Default: False.
        pretrained (str, optional): model pretrained path. Default: None
        out_stride (int): the output stride of the output feature map
    """
    # Parameters to build layers. 3 parameters are needed to construct a
    # layer, from left to right: expand_ratio, channel, num_blocks.
    arch_settings = [[1, 16, 1], [6, 24, 2], [6, 32, 3], [6, 64, 4],
                     [6, 96, 3], [6, 160, 3], [6, 320, 1]]

    def __init__(self, widen_factor=1., strides=(1, 2, 2, 2, 1, 2, 1), dilations=(1, 1, 1, 1, 1, 1, 1),
                 out_indices=(1, 2, 4, 6), frozen_stages=-1, norm_eval=False, pretrained=None,
                 progress=True, out_stride=0):
        super(MobileNetV2Encoder, self).__init__()
        self.pretrained = pretrained
        self.widen_factor = widen_factor
        self.strides = strides
        self.dilations = dilations
        assert len(strides) == len(dilations) == len(self.arch_settings)
        self.out_indices = out_indices
        for index in out_indices:
            if index not in range(0, 7):
                raise ValueError('the item in out_indices must in range(0, 7). But received {index}')
        if frozen_stages not in range(-1, 7):
            raise ValueError('frozen_stages must be in range(-1, 7). But received {frozen_stages}')
        self.out_indices = out_indices
        self.frozen_stages = frozen_stages
        self.norm_eval = norm_eval
        self.out_stride = out_stride
        self.in_channels = make_divisible(32 * widen_factor, 8)
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=self.in_channels, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(self.in_channels),
            nn.ReLU6()
        )
        self.layers = []
        for i, layer_cfg in enumerate(self.arch_settings):
            expand_ratio, channel, num_blocks = layer_cfg
            stride = self.strides[i]
            dilation = self.dilations[i]
            out_channels = make_divisible(channel * widen_factor, 8)
            inverted_res_layer = self.make_layer(
                out_channels=out_channels,
                num_blocks=num_blocks,
                stride=stride,
                dilation=dilation,
                expand_ratio=expand_ratio)
            layer_name = f'layer{i + 1}'
            self.add_module(layer_name, inverted_res_layer)
            self.layers.append(layer_name)

        if self.pretrained is None:
            self.weight_initialization()
        else:
            self.load_pretrained(progress=progress)

    def load_pretrained(self, progress):
        state_dict = load_state_dict_from_url(self.pretrained, progress=progress)
        self_state_dict = self.state_dict()
        self_keys = list(self_state_dict.keys())
        for i, (_, v) in enumerate(state_dict.items()):
            if i > len(self_keys) - 1:
                break
            self_state_dict[self_keys[i]] = v
        self.load_state_dict(self_state_dict)

    def weight_initialization(self):
        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def make_layer(self, out_channels, num_blocks, stride, dilation,
                   expand_ratio):
        """Stack InvertedResidual blocks to build a layer for MobileNetV2.
        Args:
            out_channels (int): out_channels of block.
            num_blocks (int): Number of blocks.
            stride (int): Stride of the first block.
            dilation (int): Dilation of the first block.
            expand_ratio (int): Expand the number of channels of the
                hidden layer in InvertedResidual by this ratio.
        """
        layers = []
        for i in range(num_blocks):
            layers.append(
                InvertedResidual(
                    self.in_channels,
                    out_channels,
                    stride if i == 0 else 1,
                    expand_ratio=expand_ratio,
                    dilation=dilation if i == 0 else 1)
            )
            self.in_channels = out_channels

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)

        outs = []
        for i, layer_name in enumerate(self.layers):
            layer = getattr(self, layer_name)
            x = layer(x)
            if i in self.out_indices:
                outs.append(x)

        if len(outs) == 1:
            return outs[0]
        else:
            return tuple(outs)

    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            for param in self.conv1.parameters():
                param.requires_grad = False
        for i in range(1, self.frozen_stages + 1):
            layer = getattr(self, f'layer{i}')
            layer.eval()
            for param in layer.parameters():
                param.requires_grad = False
