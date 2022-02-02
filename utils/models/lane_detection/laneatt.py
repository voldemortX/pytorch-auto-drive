import math

import torch
import numpy as np
import torch.nn as nn
from collections import OrderedDict
from ...csrc.apis import line_nms


from ..builder import MODELS


@MODELS.register()
class LaneAtt(nn.Module):
    def __init__(self,
                 backbone_cfg,
                 backbone_channels,
                 backbone_os,
                 num_points=72,
                 img_w=640,
                 img_h=360,
                 ori_img_w=1640,
                 ori_img_h=590,
                 topk_anchors=None,
                 anchor_freq_path=None,
                 anchor_feat_channels=None,
                 conf_thres=None,
                 nms_thres=0,
                 nms_topk=3000):
        super().__init__()
        self.backbone = MODELS.from_dict(backbone_cfg)
        self.backbone_channels = backbone_channels
        self.stride = backbone_os
        self.num_strips = num_points - 1
        self.num_offsets = num_points
        self.img_h = img_h
        self.img_w = img_w
        self.ori_img_w = ori_img_w,
        self.ori_img_h = ori_img_h,
        self.featmap_h = img_h // self.stride
        self.featmap_w = img_w // self.stride
        self.anchor_ys = torch.linspace(1, 0, steps=self.num_offsets, dtype=torch.float32)
        self.anchor_cut_ys = torch.linspace(1, 0, steps=self.featmap_h, dtype=torch.float32)
        self.anchor_feat_channels = anchor_feat_channels
        # nms config
        self.conf_thres = conf_thres
        self.nms_thres = nms_thres
        self.nms_topk = nms_topk

        # Anchor angles, same ones used in Line-CNN
        self.left_angles = [72., 60., 49., 39., 30., 22.]
        self.right_angles = [108., 120., 131., 141., 150., 158.]
        self.bottom_angles = [165., 150., 141., 131., 120., 108., 100., 90., 80., 72., 60., 49., 39., 30., 15.]

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
        anchors = torch.zeros((n_anchors, 2 + 2 + 1 + self.num_offsets))
        anchors_cut = torch.zeros((n_anchors, 2 + 2 + 1 + self.featmap_h))
        for i, start in enumerate(starts):
            for j, angle in enumerate(angles):
                k = i * len(angles) + j
                anchors[k] = self.generate_anchor(start, angle)
                anchors_cut[k] = self.generate_anchor(start, angle, cut=True)

        return anchors, anchors_cut

    def generate_anchor(self, start, angle, cut=False):
        if cut:
            anchor_ys = self.anchor_cut_ys
            anchor = torch.zeros(2 + 2 + 1 + self.featmap_h)
        else:
            anchor_ys = self.anchor_ys
            anchor = torch.zeros(2 + 2 + 1 + self.num_offsets)
        angle = angle * math.pi / 180.  # degrees to radians
        start_x, start_y = start
        anchor[2] = 1 - start_y  # using left bottom as the (0, 0) of the axis ?
        anchor[3] = start_x
        anchor[5:] = (start_x + (1 - anchor_ys - 1 + start_y) / math.tan(angle)) * self.img_w

        return anchor

    def compute_anchor_cut_indices(self, num_channels, feat_w, feat_h):
        # definitions
        num_proposal = len(self.anchors_cut)

        # indexing
        # num_anchors x feat_h
        unclamped_xs = torch.flip((self.anchors_cut[:, 5:] / self.stride).round().long(), dims=(1, ))
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
        # print('anchor shape {}'.format(self.anchors.shape))
        reg_proposals = torch.zeros((*cls_logits.shape[:2], 5 + self.num_offsets), device=x.device)
        # print('reg_proposal {}'.format(reg_proposals.shape))
        reg_proposals += self.anchors
        reg_proposals[:, :, :2] = cls_logits
        reg_proposals[:, :, 4:] += reg
        # # Apply nms
        # proposals_list = self.nms(reg_proposals, attention_matrix, nms_thres, nms_topk, conf_thres)

        out = OrderedDict()
        out['proposals_list'] = reg_proposals

        return out

    @torch.no_grad()
    def inference(self, inputs, forward=True):
        outputs = self.forward(inputs) if forward else inputs  # Support no forwarding inside this function
        # print(outputs['proposals_list'][0, :, 2])

        nms_outputs = self.nms(outputs['proposals_list'])
        # the number of lanes is 1 ???
        softmax = nn.Softmax(dim=1)
        decoded = []
        for proposals in nms_outputs:
            proposals[:, :2] = softmax(proposals[:, :2])
            proposals[:, 4] = torch.round(proposals[:, 4])
            if proposals.shape[0] == 0:
                decoded.append([])
                continue
            pred = self.proposals_to_pred(proposals)
            decoded.append(pred)
        return decoded

    @torch.no_grad()
    def nms(self, batch_proposals):
        softmax = nn.Softmax(dim=1)
        proposals_list = []
        for proposals in batch_proposals:
            # The gradients do not have to (and can't) be calculated for the NMS procedure
            scores = softmax(proposals[:, :2])[:, 1]
            if self.conf_threshold is not None:
                # apply confidence threshold
                above_threshold = scores > self.conf_threshold
                proposals = proposals[above_threshold]
                scores = scores[above_threshold]
            if proposals.shape[0] == 0:
                proposals_list.append((proposals[[]], self.anchors[[]]))
                continue
            keep, num_to_keep, _ = line_nms.forward(proposals, scores, self.nms_thres, self.nms_topk)
            keep = keep[:num_to_keep]
            proposals = proposals[keep]
            proposals_list.append((proposals, self.anchors[keep]))

        return proposals_list

    def proposals_to_pred(self, proposals):
        self.anchor_ys = self.anchor_ys.to(proposals.device)
        self.anchor_ys = self.anchor_ys.double()
        lanes = []
        for lane in proposals:
            lane_xs = lane[5:] / self.img_w
            start = int(round(lane[2].item() * self.num_strips))
            length = int(round(lane[4].item()))
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
            lane_xs = lane_xs.flip(0).double()
            lane_ys = lane_ys.flip(0)
            if len(lane_xs) <= 1:
                continue
            points = torch.stack((lane_xs.reshape(-1, 1), lane_ys.reshape(-1, 1)), dim=1).squeeze(2)
            points = points.cpu().numpy()
            lane_coords = []
            for i in range(points.shape[0]):
                lane_coords.append([points[i, 0] * float(self.ori_img_w[0]), points[i, 1] * float(self.ori_img_h[0])])
            # print(lane_coords)
            lanes.append(lane_coords)

        return lanes
