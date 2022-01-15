import torch
import numpy as np

from .utils import lane_pruning
from ...curve_utils import BezierCurve


class BezierBaseNet(torch.nn.Module):
    def __init__(self, thresh=0.5):
        super().__init__()
        self.thresh = thresh

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    @staticmethod
    def bezier_to_coordinates(control_points, existence, resize_shape, dataset, bezier_curve, ppl=56, gap=10):
        # control_points: L x N x 2
        H, W = resize_shape
        cps_of_lanes = []
        for flag, cp in zip(existence, control_points):
            if flag:
                cps_of_lanes.append(cp.tolist())
        coordinates = []
        for cps_of_lane in cps_of_lanes:
            bezier_curve.assign_control_points(cps_of_lane)
            if dataset == 'tusimple':
                # Find x for TuSimple's fixed y eval positions (suboptimal)
                bezier_threshold = 5.0 / H
                h_samples = np.array([1.0 - (ppl - i) * gap / H for i in range(ppl)], dtype=np.float32)
                sampled_points = bezier_curve.quick_sample_point(image_size=None)
                temp = []
                dis = np.abs(np.expand_dims(h_samples, -1) - sampled_points[:, 1])
                idx = np.argmin(dis, axis=-1)
                for i in range(ppl):
                    h = H - (ppl - i) * gap
                    if dis[i][idx[i]] > bezier_threshold or sampled_points[idx[i]][0] > 1 or sampled_points[idx[i]][0] < 0:
                        temp.append([-2, h])
                    else:
                        temp.append([sampled_points[idx[i]][0] * W, h])
                coordinates.append(temp)
            elif dataset in ['culane', 'llamas']:
                temp = bezier_curve.quick_sample_point(image_size=None)
                temp[:, 0] = temp[:, 0] * W
                temp[:, 1] = temp[:, 1] * H
                coordinates.append(temp.tolist())
            else:
                raise ValueError

        return coordinates

    @torch.no_grad()
    def inference(self, inputs, input_sizes, gap, ppl, dataset, max_lane=0, forward=True, k=9, return_cps=False, n=50):
        outputs = self.forward(inputs) if forward else inputs  # Support no forwarding inside this function
        existence_conf = outputs['logits'].sigmoid()
        existence = existence_conf > self.thresh

        # Test local maxima
        if k > 0:
            _, max_indices = torch.nn.functional.max_pool1d(existence_conf.unsqueeze(1),
                                                            kernel_size=k, stride=1,
                                                            padding=(k - 1) // 2, return_indices=True)
            max_indices = max_indices.squeeze(1)  # B x Q
            indices = torch.arange(0, existence_conf.shape[1],
                                   dtype=existence_conf.dtype,
                                   device=existence_conf.device).unsqueeze(0).expand_as(max_indices)
            local_maxima = max_indices == indices
            existence *= local_maxima

        control_points = outputs['curves']
        if max_lane != 0:  # Lane max number prior for testing
            existence, _ = lane_pruning(existence, existence_conf, max_lane=max_lane)

        if return_cps:
            image_size = torch.tensor([input_sizes[1][1], input_sizes[1][0]],
                                      dtype=torch.float32, device=control_points.device)
            cps = control_points * image_size
            cps = [cps[i][existence[i]].cpu().numpy() for i in range(existence.shape[0])]

        existence = existence.cpu().numpy()
        control_points = control_points.cpu().numpy()
        H, _ = input_sizes[1]
        b = BezierCurve(order=3, num_sample_points=H if dataset == 'tusimple' else n)

        lane_coordinates = []
        for j in range(existence.shape[0]):
            lane_coordinates.append(self.bezier_to_coordinates(control_points=control_points[j], existence=existence[j],
                                                               resize_shape=input_sizes[1], dataset=dataset,
                                                               bezier_curve=b, gap=gap, ppl=ppl))
        if return_cps:
            return cps, lane_coordinates
        else:
            return lane_coordinates
