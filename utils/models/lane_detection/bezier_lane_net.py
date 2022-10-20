import torch
import torch.nn as nn
if torch.__version__ >= '1.6.0':
    from torch.cuda.amp import autocast
else:
    from utils.torch_amp_dummy import autocast

from .bezier_base import BezierBaseNet
from ..builder import MODELS


@MODELS.register()
class BezierLaneNet(BezierBaseNet):
    # Curve regression network, similar design as simple object detection (e.g. FCOS)
    def __init__(self,
                 backbone_cfg,
                 reducer_cfg,
                 dilated_blocks_cfg,
                 feature_fusion_cfg,
                 head_cfg,
                 aux_seg_head_cfg,
                 image_height=360,
                 num_regression_parameters=8,
                 thresh=0.5,
                 local_maximum_window_size=9):
        super(BezierLaneNet, self).__init__(thresh, local_maximum_window_size)
        global_stride = 16
        branch_channels = 256

        self.backbone = MODELS.from_dict(backbone_cfg)
        self.reducer = MODELS.from_dict(reducer_cfg)
        self.dilated_blocks = MODELS.from_dict(dilated_blocks_cfg)
        self.simple_flip_2d = MODELS.from_dict(feature_fusion_cfg)  # Name kept for legacy weights
        self.aggregator = nn.AvgPool2d(kernel_size=((image_height - 1) // global_stride + 1, 1), stride=1, padding=0)
        self.regression_head = MODELS.from_dict(head_cfg)  # Name kept for legacy weights
        self.proj_classification = nn.Conv1d(branch_channels, 1, kernel_size=1, bias=True, padding=0)
        self.proj_regression = nn.Conv1d(branch_channels, num_regression_parameters,
                                         kernel_size=1, bias=True, padding=0)
        self.segmentation_head = MODELS.from_dict(aux_seg_head_cfg)

    def forward(self, x):
        # Return shape: B x Q, B x Q x N x 2
        x = self.backbone(x)
        if isinstance(x, dict):
            x = x['out']

        if self.reducer is not None:
            x = self.reducer(x)

        # Segmentation task
        if self.segmentation_head is not None:
            segmentations = self.segmentation_head(x)
        else:
            segmentations = None

        if self.dilated_blocks is not None:
            x = self.dilated_blocks(x)

        with autocast(False):  # TODO: Support fp16 like mmcv
            x = self.simple_flip_2d(x.float())
        x = self.aggregator(x)[:, :, 0, :]

        x = self.regression_head(x)
        logits = self.proj_classification(x).squeeze(1)
        curves = self.proj_regression(x)

        return {'logits': logits,
                'curves': curves.permute(0, 2, 1).reshape(curves.shape[0], -1, curves.shape[-2] // 2, 2).contiguous(),
                'segmentations': segmentations}

    def eval(self, profiling=False):
        super().eval()
        if profiling:
            self.segmentation_head = None
