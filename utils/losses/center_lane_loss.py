import torch
from torch import Tensor
from typing import Optional
from torch.nn import functional as F

from ._utils import WeightedLoss
from .builder import LOSSES


# Typical lane detection loss by binary segmentation (e.g. SCNN)
@LOSSES.register()
class CenterLaneLoss(WeightedLoss):
    __constants__ = ['ignore_index', 'reduction']
    ignore_index: int

    def __init__(self, existence_weight: float = 0.1, weight: Optional[Tensor] = None, size_average=None,
                 ignore_index: int = -100, reduce=None, center_weight: float = 1.0, reduction: str = 'mean'):
        super(CenterLaneLoss, self).__init__(weight, size_average, reduce, reduction)
        self.ignore_index = ignore_index
        self.existence_weight = existence_weight
        self.center_weight = center_weight

    def center_loss(self, inputs: Tensor, targets: Tensor):
        binary_map = targets.clone()
        binary_map[binary_map == 255] = 0.
        binary_map[binary_map != 0] = 1.
        B, H, W = targets.size()
        coords = torch.stack(torch.meshgrid([torch.arange(H), torch.arange(W)])).to(inputs.device)  # 2, Wh, Ww
        coords_h = (coords[0][None, :, :].expand(B, H, W).float() / H) * binary_map
        coords_w = (coords[1][None, :, :].expand(B, H, W).float() / W) * binary_map
        avg_h, avg_w = self.get_avg_coords(coords_h, binary_map), self.get_avg_coords(coords_w, binary_map)
        avg_coord = torch.stack([avg_h, avg_w]).permute(1, 0).contiguous()
        coord_affinity = self.compute_self_affinity(avg_coord)
        avg_feat = F.adaptive_avg_pool2d(inputs, (1, 1)).squeeze()
        feat_affinity = self.compute_self_affinity(avg_feat)
        loss = F.l1_loss(feat_affinity, coord_affinity)
        return loss

    def get_avg_coords(self, x, mask):
        avg_coords = torch.sum(x, dim=(1, 2)) / (torch.sum(mask, dim=(1, 2)) + 1.)
        return avg_coords

    def compute_self_affinity(self, x, eps=1e-7):
        norm_x = torch.norm(x, dim=1, keepdim=True)
        self_affinity = torch.mm(x, x.t())/(torch.mm(norm_x, norm_x.t()) + eps)
        return self_affinity

    def forward(self, inputs: Tensor, targets: Tensor, lane_existence: Tensor, net, interp_size):
        outputs = net(inputs)
        prob_maps = torch.nn.functional.interpolate(outputs['out'], size=interp_size, mode='bilinear',
                                                    align_corners=True)

        targets[targets > lane_existence.shape[-1]] = 255  # Ignore extra lanes
        affinity_loss = self.center_loss(outputs['feat_map'], targets)
        segmentation_loss = F.cross_entropy(prob_maps, targets, weight=self.weight,
                                            ignore_index=self.ignore_index, reduction=self.reduction)
        existence_loss = F.binary_cross_entropy_with_logits(outputs['lane'], lane_existence,
                                                            weight=None, pos_weight=None, reduction=self.reduction)
        total_loss = segmentation_loss + self.existence_weight * existence_loss + self.center_weight * affinity_loss

        return total_loss, {'training loss': total_loss, 'loss seg': segmentation_loss,
                            'loss exist': existence_loss, 'loss affinity': affinity_loss}



