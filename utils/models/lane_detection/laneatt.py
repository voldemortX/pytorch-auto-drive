# Refactored from lucastabelini/LaneATT
# Diffs:
# 1. we changed lane rep to 74 numbers (start, len, 72 offsets)
# 2. we use a cleaner line nms dynamically loaded (input only 74 numbers, not 77)
# 3. we removed unnecessary inputs & outputs in post-processing funcs
# 4. we removed B-Spline interpolation post-processing to provide fair comparisons (test 74.88 for resnet18-CULane in official code)
# After these simplifications & refactors, our testing procedure seems even better than the original

import math
import torch
import numpy as np
import torch.nn as nn
from torch.nn import functional as F
from scipy.interpolate import splprep, splev
from .._utils import is_tracing

try:
    from ...csrc.apis import line_nms
    print('Successfully complied line nms for LaneATT.')
except:
    from ...common import warnings
    warnings.warn('Can\'t complie line nms op for LaneATT. Set verbose=True for load in /utils/csrc/apis.py L9 for details.')
from ..builder import MODELS


@MODELS.register()
class LaneAtt(nn.Module):
    # Anchor angles, same ones used in Line-CNN
    left_angles = [72., 60., 49., 39., 30., 22.]
    right_angles = [108., 120., 131., 141., 150., 158.]
    bottom_angles = [165., 150., 141., 131., 120., 108., 100., 90., 80., 72., 60., 49., 39., 30., 15.]

    def __init__(self,
                 backbone_cfg,
                 backbone_channels,
                 backbone_os,
                 num_points=72,
                 img_w=640,
                 img_h=360,
                 topk_anchors=None,
                 anchor_freq_path=None,
                 anchor_feat_channels=None,
                 conf_thres=None,
                 nms_thres=0,
                 nms_topk=3000,
                 trace_arg=None):
        super().__init__()
        self.backbone = MODELS.from_dict(backbone_cfg)
        self.backbone_channels = backbone_channels
        self.stride = backbone_os
        self.num_strips = num_points - 1
        self.num_offsets = num_points
        self.img_h = img_h
        self.img_w = img_w
        self.featmap_h = img_h // self.stride
        self.featmap_w = img_w // self.stride
        self.anchor_ys = torch.linspace(1, 0, steps=self.num_offsets, dtype=torch.float32)
        self.anchor_cut_ys = torch.linspace(1, 0, steps=self.featmap_h, dtype=torch.float32)
        self.anchor_feat_channels = anchor_feat_channels
        # nms config
        self.conf_thres = conf_thres
        self.nms_thres = nms_thres
        self.nms_topk = nms_topk

        if trace_arg is not None:  # Pre-compute
            attention_matrix = torch.eye(topk_anchors).repeat(trace_arg['bs'], 1, 1)
            self.pre_non_diag_inds = torch.nonzero(attention_matrix == 0., as_tuple=False)

        # generate anchors
        self.anchors, self.anchors_cut = self.generate_anchors(lateral_n=72, bottom_n=128)
        # Filter masks if `anchors_freq_path` is provided
        if anchor_freq_path is not None:
            anchors_mask = torch.load(anchor_freq_path).cpu()
            assert topk_anchors is not None, 'topk_anchors cannot be None'
            idx = torch.argsort(anchors_mask, descending=True)[: topk_anchors]
            self.anchors = self.anchors[idx]
            self.anchors_cut = self.anchors_cut[idx]

        # pre compute indices for the anchor pooling
        self.cut_zs, self.cut_ys, self.cut_xs, self.invalid_mask = self.compute_anchor_cut_indices(
            self.anchor_feat_channels, self.featmap_w, self.featmap_h)

        # Setup ans initialize layers
        self.conv1 = nn.Conv2d(self.backbone_channels, self.anchor_feat_channels, kernel_size=1)
        self.cls_layer = nn.Linear(2 * self.anchor_feat_channels * self.featmap_h, 2)
        self.reg_layer = nn.Linear(2 * self.anchor_feat_channels * self.featmap_h, self.num_offsets + 1)
        self.attention_layer = nn.Linear(self.anchor_feat_channels * self.featmap_h, len(self.anchors) - 1)
        self.initialize_layer(self.attention_layer)
        self.initialize_layer(self.conv1)
        self.initialize_layer(self.cls_layer)
        self.initialize_layer(self.reg_layer)

    def generate_anchors(self, lateral_n, bottom_n):
        left_anchors, left_cut = self.generate_side_anchors(self.left_angles, x=0., nb_origins=lateral_n)
        right_anchors, right_cut = self.generate_side_anchors(self.right_angles, x=1., nb_origins=lateral_n)
        bottom_anchors, bottom_cut = self.generate_side_anchors(self.bottom_angles, y=1., nb_origins=bottom_n)

        return torch.cat([left_anchors, bottom_anchors, right_anchors]), \
            torch.cat([left_cut, bottom_cut, right_cut])

    def generate_side_anchors(self, angles, nb_origins, x=None, y=None):
        if x is None and y is not None:
            starts = [(x, y) for x in np.linspace(1., 0., num=nb_origins)]
        elif x is not None and y is None:
            starts = [(x, y) for y in np.linspace(1., 0., num=nb_origins)]
        else:
            raise Exception('Please define exactly one of `x` or `y` (not neither nor both)')

        n_anchors = nb_origins * len(angles)

        # each row, first for x and second for y:
        # 2 scores, 1 start_y, start_x, 1 length, num_points coordinates
        # score[0] = negative prob, score[0] = positive prob
        anchors = torch.zeros((n_anchors, 2 + self.num_offsets))
        anchors_cut = torch.zeros((n_anchors, 2 + self.featmap_h))
        for i, start in enumerate(starts):
            for j, angle in enumerate(angles):
                k = i * len(angles) + j
                anchors[k] = self.generate_anchor(start, angle)
                anchors_cut[k] = self.generate_anchor(start, angle, cut=True)

        return anchors, anchors_cut

    def generate_anchor(self, start, angle, cut=False):
        if cut:
            anchor_ys = self.anchor_cut_ys
            anchor = torch.zeros(2 + self.featmap_h)
        else:
            anchor_ys = self.anchor_ys
            anchor = torch.zeros(2 + self.num_offsets)
        angle = angle * math.pi / 180.  # degrees to radians
        start_x, start_y = start
        anchor[0] = 1 - start_y  # using left bottom as the (0, 0) of the axis ?
        anchor[1] = start_x
        anchor[2:] = (start_x + (1 - anchor_ys - 1 + start_y) / math.tan(angle)) * self.img_w

        return anchor

    def compute_anchor_cut_indices(self, num_channels, feat_w, feat_h):
        # definitions
        num_proposal = len(self.anchors_cut)

        # indexing
        # num_anchors x feat_h
        unclamped_xs = torch.flip((self.anchors_cut[:, 2:] / self.stride).round().long(), dims=(1, ))
        unclamped_xs = unclamped_xs[..., None]
        # num_channels x num_anchors x feat_h --> num_channels * num_anchors * feat_h x 1
        unclamped_xs = torch.repeat_interleave(unclamped_xs, num_channels, dim=0).reshape(-1, 1)
        cut_xs = torch.clamp(unclamped_xs, 0, feat_w - 1)
        unclamped_xs = unclamped_xs.reshape(num_proposal, num_channels, feat_h, 1)
        invalid_mask = (unclamped_xs < 0) | (unclamped_xs > feat_w)
        cut_ys = torch.arange(0, feat_h)
        cut_ys = cut_ys.repeat(num_channels * num_proposal)[:, None].reshape(num_proposal, num_channels, feat_h)
        cut_ys = cut_ys.reshape(-1, 1)
        cut_zs = torch.arange(num_channels).repeat_interleave(feat_h).repeat(num_proposal)[:, None]

        return cut_zs, cut_ys, cut_xs, invalid_mask

    def cut_anchor_features(self, features):
        # definitions
        batch_size = features.shape[0]
        n_proposals = len(self.anchors)
        n_fmaps = features.shape[1]
        batch_anchor_features = torch.zeros((batch_size, n_proposals, n_fmaps, self.featmap_h, 1),
                                            device=features.device)

        # actual cutting
        for batch_idx, img_features in enumerate(features):
            rois = img_features[self.cut_zs, self.cut_ys, self.cut_xs].view(n_proposals, n_fmaps, self.featmap_h, 1)
            rois[self.invalid_mask] = 0
            batch_anchor_features[batch_idx] = rois

        return batch_anchor_features

    @staticmethod
    def initialize_layer(layer):
        if isinstance(layer, (nn.Conv2d, nn.Linear)):
            torch.nn.init.normal_(layer.weight, mean=0., std=0.001)
            if layer.bias is not None:
                torch.nn.init.constant_(layer.bias, 0)

    def cuda(self, device=None):
        cuda_self = super().cuda(device)
        cuda_self.anchors = cuda_self.anchors.cuda(device)
        cuda_self.anchor_ys = cuda_self.anchor_ys.cuda(device)
        cuda_self.cut_zs = cuda_self.cut_zs.cuda(device)
        cuda_self.cut_ys = cuda_self.cut_ys.cuda(device)
        cuda_self.cut_xs = cuda_self.cut_xs.cuda(device)
        cuda_self.invalid_mask = cuda_self.invalid_mask.cuda(device)
        return cuda_self

    def to(self, *args, **kwargs):
        device_self = super().to(*args, **kwargs)
        device_self.anchors = device_self.anchors.to(*args, **kwargs)
        device_self.anchor_ys = device_self.anchor_ys.to(*args, **kwargs)
        device_self.cut_zs = device_self.cut_zs.to(*args, **kwargs)
        device_self.cut_ys = device_self.cut_ys.to(*args, **kwargs)
        device_self.cut_xs = device_self.cut_xs.to(*args, **kwargs)
        device_self.invalid_mask = device_self.invalid_mask.to(*args, **kwargs)
        return device_self

    def forward(self, x):
        batch_features = self.backbone(x)['out']

        batch_features = self.conv1(batch_features)
        # batchsize x num_proposals x anchor_feat_channels x featmap_h x 1
        batch_anchor_features = self.cut_anchor_features(batch_features)

        # join proposals from all images into a single proposals features batch
        batch_anchor_features = batch_anchor_features.view(-1, self.anchor_feat_channels * self.featmap_h)
        # add attention features
        softmax = nn.Softmax(dim=1)
        scores = self.attention_layer(batch_anchor_features)
        attention = softmax(scores).reshape(x.shape[0], len(self.anchors), -1)
        attention_matrix = torch.eye(attention.shape[1], device=x.device).repeat(x.shape[0], 1, 1)
        if is_tracing():
            # Use pre-computed nonzero results
            non_diag_inds = self.pre_non_diag_inds.to(attention_matrix.device)
        else:
            non_diag_inds = torch.nonzero(attention_matrix == 0., as_tuple=False)
        attention_matrix[:] = 0
        attention_matrix[non_diag_inds[:, 0], non_diag_inds[:, 1], non_diag_inds[:, 2]] = attention.flatten()
        batch_anchor_features = batch_anchor_features.reshape(x.shape[0], len(self.anchors), -1)
        attention_features = torch.bmm(torch.transpose(batch_anchor_features, 1, 2),
                                       torch.transpose(attention_matrix, 1, 2)).transpose(1, 2)
        attention_features = attention_features.reshape(-1, self.anchor_feat_channels * self.featmap_h)
        batch_anchor_features = batch_anchor_features.reshape(-1, self.anchor_feat_channels * self.featmap_h)
        batch_anchor_features = torch.cat((attention_features, batch_anchor_features), dim=1)

        # predict
        cls_logits = self.cls_layer(batch_anchor_features)
        reg = self.reg_layer(batch_anchor_features)

        # Undo joining
        cls_logits = cls_logits.reshape(x.shape[0], -1, cls_logits.shape[1])
        reg = reg.reshape(x.shape[0], -1, reg.shape[1])

        # Add offset to anchors
        reg_proposals = torch.zeros((*cls_logits.shape[:2], self.num_offsets + 2), device=x.device)
        reg_proposals += self.anchors
        reg_proposals[:, :, 1:] += reg

        return {
            'offsets': reg_proposals[..., 2:],  # B x M x 72
            'starts': reg_proposals[..., 0],  # B x M x 1
            'lengths': reg_proposals[..., 1],  # B x M x 1
            'logits': cls_logits  # B x M x 2
        }

    @torch.no_grad()
    def inference(self, inputs, input_sizes, forward=True, *args, **kwargs):
        outputs = self.forward(inputs) if forward else inputs  # Support no forwarding inside this function
        to_tusimple = True if args[1] == "tusimple" else False
        batch_regs = self.nms(outputs)
        # the number of lanes is 1 ???
        decoded = []
        for regs in batch_regs:
            regs[:, 1] = torch.round(regs[:, 1])  # length
            if regs.shape[0] == 0:
                decoded.append([])
                continue
            pred = self.proposals_to_pred(regs, input_sizes[1], to_tusimple)
            decoded.append(pred)
        return decoded

    @torch.no_grad()
    def nms(self, batch_proposals):
        proposals_list = []
        for i in range(len(batch_proposals['logits'])):
            scores = F.softmax(batch_proposals['logits'][i], dim=1)[:, 1]
            regs = torch.cat([batch_proposals['starts'][i][..., None],
                              batch_proposals['lengths'][i][..., None],
                              batch_proposals['offsets'][i]], dim=1)
            if self.conf_thres is not None:
                # apply confidence threshold
                above_threshold = scores > self.conf_thres
                regs = regs[above_threshold]
                scores = scores[above_threshold]
            if regs.shape[0] == 0:
                proposals_list.append(regs[[]])
                continue
            keep, num_to_keep, _ = line_nms(regs, scores, self.nms_thres, self.nms_topk)
            keep = keep[:num_to_keep]
            regs = regs[keep]
            proposals_list.append(regs)

        return proposals_list

    def proposals_to_pred(self, proposals, image_size, to_tusimple=False):
        self.anchor_ys = self.anchor_ys.to(proposals.device)
        lanes = []
        for lane in proposals:
            lane_xs = lane[2:] / self.img_w

            # start end length in 0-72
            start = int(round(lane[0].item() * self.num_strips))
            length = int(round(lane[1].item()))
            end = start + length - 1
            end = min(end, len(self.anchor_ys) - 1)

            # end = label_end
            # if the proposal does not start at the bottom of the image,
            # extend its proposal until the x is outside the image
            mask = ~((((lane_xs[:start] >= 0.) &
                       (lane_xs[:start] <= 1.)).cpu().numpy()[::-1].cumprod()[::-1]).astype(np.bool))
            lane_xs[end + 1:] = -2
            lane_xs[:start][mask] = -2
            lane_ys = self.anchor_ys[lane_xs >= 0]
            lane_xs = lane_xs[lane_xs >= 0]
            lane_xs = lane_xs.flip(0)
            lane_ys = lane_ys.flip(0)
            if len(lane_xs) <= 1:
                continue
            points = torch.stack((lane_xs.reshape(-1, 1), lane_ys.reshape(-1, 1)), dim=1).squeeze(2)
            points = points.cpu().numpy()
            lane_coords = []
            for i in range(points.shape[0]):
                lane_coords.append([points[i, 0] * float(image_size[1]), points[i, 1] * float(image_size[0])])
            if to_tusimple:
                lanes.append(self.convert_to_tusimple(lane_coords))
            else:
                lanes.append(lane_coords)

        return lanes

    def convert_to_tusimple(self, points, n=200, bezier_threshold=5):
        """Spline interpolation of a lane. Used on the predictions"""
        x = [x for x, _ in points]
        y = [y for _, y in points]
        tck, _ = splprep([x, y], s=0, t=n, k=min(3, len(points) - 1))
        u = np.linspace(0., 1., n)
        rep_points = np.array(splev(u, tck)).T
        h_samples = [(160 + y * 10) for y in range(56)]
        temp = []
        for h_sample in h_samples:
            dis = np.abs(h_sample - rep_points[:, 1])
            idx = np.argmin(dis)
            if dis[idx] > bezier_threshold:
                temp.append([-2, h_sample])
            else:
                temp.append([round(rep_points[:, 0][idx], 3), h_sample])
        return temp
