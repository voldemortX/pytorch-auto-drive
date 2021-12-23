# Adapted from ZJULearning/resa
# Better to use a decoupled implementation,
# costs more codes, but clear.
# Diff from RESA official code:
# 1. we use BN+ReLU in channel reducer
# 2. we always use the BUSD decoder in the paper (official code does not use BUSD in CULane)
# 3. we always use 5 RESA iterations (4 in official code)
# 4. we use a higher capacity lane existence classifier (same as ERFNet/ENet baseline)
# 5. we use the SCNN sqrt(5) init trick for RESA, which
# 5.1. enables fewer warmup steps
# 5.2. combined with 4, produces slightly better performance
# 6. we do not use horizontal flip or cutting height in loading, in which
# 6.1. flip does not help performance (at least on the val set)
# 6.2. w.o. cutting height trick probably is the main reason for our lower performance, but we can't use it since
# other pytorch-auto-drive models do not use it.
import torch.nn as nn

from ..builder import MODELS


@MODELS.register()
class RESA_Net(nn.Module):
    def __init__(self,
                 backbone_cfg,
                 reducer_cfg,
                 spatial_conv_cfg,
                 classifier_cfg,
                 lane_classifier_cfg,
                 trace_arg=None):
        super().__init__()
        self.backbone = MODELS.from_dict(backbone_cfg)
        # self.channel_reducer = RESAReducer(in_channels=in_channels, reduce=channel_reduce, bn_relu=False)
        self.channel_reducer = MODELS.from_dict(reducer_cfg)
        self.spatial_conv = MODELS.from_dict(spatial_conv_cfg, trace_arg=trace_arg)
        self.decoder = MODELS.from_dict(classifier_cfg)
        # self.decoder = PlainDecoder(num_classes=num_classes)
        self.lane_classifier = MODELS.from_dict(lane_classifier_cfg)
        # self.lane_classifier = RESALaneExist(num_output=num_classes - 1, flattened_size=flattened_size)

    def forward(self, x):
        x = self.backbone(x)['out']
        x = self.channel_reducer(x)
        x = self.spatial_conv(x)

        res = {'out': self.decoder(x),
               'lane': self.lane_classifier(x)}

        return res
