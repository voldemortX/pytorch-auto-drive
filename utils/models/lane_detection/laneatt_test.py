# Demo class
import torch
import torch.nn as nn

from ..builder import MODELS
from ...common import warnings
try:
    from ...csrc.apis import line_nms
    print('Successfully complied line nms for LaneATT.')
except:
    warnings.warn('Can\'t complie line nms op for LaneATT.')


@MODELS.register()
class LaneATT_Test(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self):
        pass

    @torch.no_grad()
    def inference(self, proposals, scores, nms_thres, nms_topk):
        return line_nms(proposals, scores, nms_thres, nms_topk)
