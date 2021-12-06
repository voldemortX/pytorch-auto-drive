# Adapted from liuruijin17/LSTR
import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils import lane_pruning
from .. import resnet
from ..transformer import build_transformer, build_position_encoding
from ..resnet import resnet18_reduced
from ..mlp import MLP
from .._utils import IntermediateLayerGetter, is_tracing


def cubic_curve_with_projection(coefficients, y):
    # The cubic curve model from LSTR (considers projection to image plane)
    # Return x coordinates
    # coefficients: [d1, d2, ..., 6]
    # 6 coefficients: [k", f", m", n", b", b''']
    # y: [d1, d2, ..., N]
    y = y.permute(-1, *[i for i in range(len(y.shape) - 1)])  # -> [N, d1, d2, ...]
    x = coefficients[..., 0] / (y - coefficients[..., 1]) ** 2 \
        + coefficients[..., 2] / (y - coefficients[..., 1]) \
        + coefficients[..., 3] \
        + coefficients[..., 4] * y \
        - coefficients[..., 5]

    return x.permute(*[i + 1 for i in range(len(x.shape) - 1)], 0)  # [d1, d2, ... , N]


class LSTR(nn.Module):
    def __init__(self,
                 expansion=1,  # Expansion rate (1x for TuSimple & 2x for CULane)
                 num_queries=7,  # Maximum number of lanes
                 aux_loss=True,  # Important for transformer-based methods
                 pos_type='sine',
                 drop_out=0.1,
                 num_heads=2,
                 enc_layers=2,
                 dec_layers=2,
                 pre_norm=False,
                 return_intermediate=True,
                 lsp_dim=8,
                 mlp_layers=3,
                 backbone_name='resnet18s',
                 trace_arg=None
                 ):
        super(LSTR, self).__init__()

        if backbone_name == 'resnet18s':  # Original LSTR backbone
            backbone = resnet18_reduced(
                pretrained=False, expansion=expansion,
                replace_stride_with_dilation=[False, False, False])
        else:  # Common backbones
            backbone = resnet.__dict__[backbone_name](
                pretrained=True,
                replace_stride_with_dilation=[False, False, False])

        return_layers = {'layer4': 'out'}
        self.backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)

        hidden_dim = 32 * expansion
        self.aux_loss = aux_loss
        self.position_embedding = build_position_encoding(hidden_dim=hidden_dim, position_embedding=pos_type)
        if trace_arg is not None:  # Pre-compute embeddings
            trace_arg['h'] = (trace_arg['h'] - 1) // 32 + 1
            trace_arg['w'] = (trace_arg['w'] - 1) // 32 + 1
            x = torch.zeros((trace_arg['bs'], trace_arg['h'], trace_arg['w']), dtype=torch.bool)
            y = torch.zeros((trace_arg['bs'], 128 * expansion, trace_arg['h'], trace_arg['w']), dtype=torch.float32)
            self.pos = torch.nn.Parameter(data=self.position_embedding(y, x), requires_grad=False)
        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        self.input_proj = nn.Conv2d(128 * expansion, hidden_dim, kernel_size=1)  # Same channel as layer4

        self.transformer = build_transformer(hidden_dim=hidden_dim,
                                             dropout=drop_out,
                                             nheads=num_heads,
                                             dim_feedforward=128 * expansion,
                                             enc_layers=enc_layers,
                                             dec_layers=dec_layers,
                                             pre_norm=pre_norm,
                                             return_intermediate_dec=return_intermediate)

        # Original LSTR: 3 classes + CE (softmax), we use 2
        self.class_embed = nn.Linear(hidden_dim, 2)

        self.specific_embed = MLP(hidden_dim, hidden_dim, lsp_dim - 4, mlp_layers)  # Specific for each lane
        self.shared_embed = MLP(hidden_dim, hidden_dim, 4, mlp_layers)  # 4 shared curve coefficients

    def forward(self, images, padding_masks=None):
        # images: B x C x H x W
        # padding_masks: B x H x W (0 or 1 -> ignored)
        p = self.backbone(images)['out']

        # Padding mask (for paddings added in transforms)
        if is_tracing():
            pos = self.pos
        else:
            if padding_masks is None:  # Make things easier for testing (assume no padding)
                padding_masks = torch.zeros((p.shape[0], p.shape[2], p.shape[3]), dtype=torch.bool, device=p.device)
            else:
                padding_masks = F.interpolate(padding_masks[None].float(), size=p.shape[-2:]).to(torch.bool)[0]
            pos = self.position_embedding(p, padding_masks)

        hs, _ = self.transformer(self.input_proj(p), padding_masks, self.query_embed.weight, pos)
        output_class = self.class_embed(hs)
        output_specific = self.specific_embed(hs)
        output_shared = self.shared_embed(hs)
        output_shared = torch.mean(output_shared, dim=-2, keepdim=True)  # Why not take mean on input and simply expand?
        output_shared = output_shared.repeat(1, 1, output_specific.shape[2], 1)

        # Keep this for consistency with official LSTR: [upper, lower, k", f", m", n", b", b''']
        output_curve = torch.cat([output_specific[:, :, :, :2],
                                  output_shared, output_specific[:, :, :, 2:]], dim=-1)

        out = {'logits': output_class[-1], 'curves': output_curve[-1]}  # Last layer result
        if self.aux_loss:
            out['aux'] = self._set_aux_loss(output_class, output_curve)  # All intermediate results

        return out

    @torch.no_grad()
    def inference(self, inputs, input_sizes, gap, ppl, dataset, max_lane=0, forward=True):
        outputs = self.forward(inputs) if forward else inputs  # Support no forwarding inside this function
        existence_conf = outputs['logits'].softmax(dim=-1)[..., 1]
        existence = outputs['logits'].max(dim=-1).indices == 1
        if max_lane != 0:  # Lane max number prior for testing
            existence, _ = lane_pruning(existence, existence_conf, max_lane=max_lane)

        existence = existence.cpu().numpy()
        # Get coordinates for lanes
        lane_coordinates = []
        for j in range(existence.shape[0]):
            lane_coordinates.append(self.coefficients_to_coordinates(outputs['curves'][j, :, 2:], existence[j],
                                    resize_shape=input_sizes[1], dataset=dataset, ppl=ppl,
                                    gap=gap, curve_function=cubic_curve_with_projection,
                                    upper_bound=outputs['curves'][j, :, 0],
                                    lower_bound=outputs['curves'][j, :, 1]))

        return lane_coordinates

    @staticmethod
    def coefficients_to_coordinates(coefficients, existence, resize_shape, dataset, ppl, gap, curve_function,
                                    upper_bound, lower_bound):
        # For methods that predict coefficients of polynomials,
        # works with normalized coordinates (in range 0.0 ~ 1.0).
        # Restricted to single image to align with other methods' codes
        H, W = resize_shape
        if dataset == 'tusimple':  # Annotation start at 10 pixel away from bottom
            y = torch.tensor([1.0 - (ppl - i) * gap / H for i in range(ppl)],
                             dtype=coefficients.dtype, device=coefficients.device)
        elif dataset in ['culane', 'llamas']:  # Annotation start at bottom
            y = torch.tensor([1.0 - i * gap / H for i in range(ppl)],
                             dtype=coefficients.dtype, device=coefficients.device)
        else:
            raise ValueError
        coords = curve_function(coefficients=coefficients, y=y.unsqueeze(0).expand(coefficients.shape[0], -1))

        # Delete outside points according to predicted upper & lower boundaries
        coordinates = []
        for i in range(existence.shape[0]):
            if existence[i]:
                # Note that in image coordinate system, (0, 0) is the top-left corner
                valid_points = (coords[i] >= 0) * (coords[i] <= 1) * (y < lower_bound[i]) * (y > upper_bound[i])
                if valid_points.sum() < 2:  # Same post-processing technique as segmentation methods
                    continue
                if dataset == 'tusimple':  # Invalid sample points need to be included as negative value, e.g. -2
                    coordinates.append([[(coords[i][j] * W).item(), H - (ppl - j) * gap]
                                        if valid_points[j] else [-2, H - (ppl - j) * gap] for j in range(ppl)])
                elif dataset in ['culane', 'llamas']:
                    coordinates.append([[(coords[i][j] * W).item(), H - j * gap]
                                        for j in range(ppl) if valid_points[j]])
                else:
                    raise ValueError

        return coordinates

    @torch.jit.unused
    def _set_aux_loss(self, output_class, output_curve):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{'logits': a, 'curves': b} for a, b in zip(output_class[:-1], output_curve[:-1])]
