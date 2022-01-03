import torch.nn as nn
from ..builder import MODELS
from ..common_models import InvertedResidualV3
import torch

@MODELS.register()
class MobileNetV3(nn.Module):
    """MobileNetV3 backbone.
    This backbone is the improved implementation of `Searching for MobileNetV3
    <https://ieeexplore.ieee.org/document/9008835>`_.
    Args:
        arch (str): Architecture of mobilnetv3, from {'small', 'large'}.
            Default: 'small'.
        out_indices (tuple[int]): Output from which layer.
            Default: (0, 1, 12).
        frozen_stages (int): Stages to be frozen (all param fixed).
            Default: -1, which means not freezing any parameters.
        norm_eval (bool): Whether to set norm layers to eval mode, namely,
            freeze running stats (mean and var). Note: Effect on Batch Norm
            and its variants only. Default: False.
        pretrained (str, optional): model pretrained path. Default: None
    """
    # MobileNet V3 for segmentation
    # Parameters to build each block:
    #     [kernel size, mid channels, out channels, with_se, act type, stride, dilated]
    arch_settings = {
        'small': [[3, 16, 16, True, 'ReLU', 2, False],  # block0 layer1 os=4
                  [3, 72, 24, False, 'ReLU', 2, False],  # block1 layer2 os=8
                  [3, 88, 24, False, 'ReLU', 1, False],
                  [5, 96, 40, True, 'HSwish', 2, False],  # block2 layer4 os=16
                  [5, 240, 40, True, 'HSwish', 1, False],
                  [5, 240, 40, True, 'HSwish', 1, False],
                  [5, 120, 48, True, 'HSwish', 1, False],  # block3 layer7 os=16
                  [5, 144, 48, True, 'HSwish', 1, False],
                  [5, 288, 96, True, 'HSwish', 2, True],  # block4 layer9 os=32
                  [5, 576, 96, True, 'HSwish', 1, True],
                  [5, 576, 96, True, 'HSwish', 1, True]],
        'large': [[3, 16, 16, False, 'ReLU', 1, False],  # block0 layer1 os=2
                  [3, 64, 24, False, 'ReLU', 2, False],  # block1 layer2 os=4
                  [3, 72, 24, False, 'ReLU', 1, False],
                  [5, 72, 40, True, 'ReLU', 2, False],  # block2 layer4 os=8
                  [5, 120, 40, True, 'ReLU', 1, False],
                  [5, 120, 40, True, 'ReLU', 1, False],
                  [3, 240, 80, False, 'HSwish', 2, False],  # block3 layer7 os=16
                  [3, 200, 80, False, 'HSwish', 1, False],
                  [3, 184, 80, False, 'HSwish', 1, False],
                  [3, 184, 80, False, 'HSwish', 1, False],
                  [3, 480, 112, True, 'HSwish', 1, False],  # block4 layer11 os=16
                  [3, 672, 112, True, 'HSwish', 1, False],
                  [5, 672, 160, True, 'HSwish', 2, True],  # block5 layer13 os=32
                  [5, 960, 160, True, 'HSwish', 1, True],
                  [5, 960, 160, True, 'HSwish', 1, True]]
    }  # yapf: disable

    def __init__(self, arch='small', out_indices=(0, 1, 12), frozen_stages=-1, reduction_factor=1,
                 norm_eval=False, pretrained=None):
        super(MobileNetV3, self).__init__()

        self.pretrained = pretrained
        assert arch in self.arch_settings
        assert isinstance(reduction_factor, int) and reduction_factor > 0

        for index in out_indices:
            if index not in range(0, len(self.arch_settings[arch]) + 2):
                raise ValueError(
                    'the item in out_indices must in '
                    f'range(0, {len(self.arch_settings[arch]) + 2}). '
                    f'But received {index}')

        if frozen_stages not in range(-1, len(self.arch_settings[arch]) + 2):
            raise ValueError('frozen_stages must be in range(-1, '
                             f'{len(self.arch_settings[arch]) + 2}). '
                             f'But received {frozen_stages}')
        self.arch = arch
        self.out_indices = out_indices
        self.frozen_stages = frozen_stages
        self.reduction_factor = reduction_factor
        self.norm_eval = norm_eval
        self.layers = self._make_layer()

        if self.pretrained is None:
            self.weight_initialization()
        else:
            self.load_pretrained()

    def _make_layer(self):
        layers = []

        # build the first layer (layer0)
        in_channels = 16
        layer = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=in_channels, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.Hardswish()
        )
        self.add_module('layer0', layer)
        layers.append('layer0')

        layer_setting = self.arch_settings[self.arch]
        for i, params in enumerate(layer_setting):
            (kernel_size, mid_channels, out_channels, with_se, act,
             stride, dilated) = params

            if self.arch == 'large' and i >= 12 or self.arch == 'small' and i >= 8:
                mid_channels = mid_channels // self.reduction_factor
                out_channels = out_channels // self.reduction_factor
            dilation = 2 if dilated else 1
            layer = InvertedResidualV3(in_channels=in_channels, out_channels=out_channels, mid_channels=mid_channels,
                                       kernel_size=kernel_size, stride=stride, with_se=with_se, act=act,
                                       with_expand_conv=(in_channels != mid_channels), dilation=dilation)
            in_channels = out_channels
            layer_name = 'layer{}'.format(i + 1)
            self.add_module(layer_name, layer)
            layers.append(layer_name)

        # build the last layer
        # block5 layer12 os=32 for small model
        # block6 layer16 os=32 for large model
        out_channels = 576 if self.arch == 'small' else 960
        layer = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, dilation=4,
                      padding=0, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.Hardswish()
        )
        layer_name = 'layer{}'.format(len(layer_setting) + 1)
        self.add_module(layer_name, layer)
        layers.append(layer_name)

        return layers

    def forward(self, x):
        outs = []
        for i, layer_name in enumerate(self.layers):
            layer = getattr(self, layer_name)
            x = layer(x)
            if i in self.out_indices:
                outs.append(x)
        return outs[-1]

    def _freeze_stages(self):
        for i in range(self.frozen_stages + 1):
            layer = getattr(self, f'layer{i}')
            layer.eval()
            for param in layer.parameters():
                param.requires_grad = False

    def load_pretrained(self):
        state_dict = torch.load(self.pretrained)
        self_state_dict = self.state_dict()
        self_keys = list(self_state_dict.keys())
        for i, (_, v) in enumerate(state_dict.items()):
            if i > len(self_keys) - 1:
                break
            self_state_dict[self_keys[i]] = v
        self.load_state_dict(self_state_dict)

    def weight_initialization(self):
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


