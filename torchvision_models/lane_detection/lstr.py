# Adapted from liuruijin17/LSTR
import torch
import torch.nn as nn
import torch.nn.functional as F
from .. import resnet
from ..transformer import build_transformer, build_position_encoding
from ..resnet import resnet18_reduced
from ..mlp import MLP
from .._utils import IntermediateLayerGetter


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
                 backbone_name='resnet18s'
                 ):
        super(LSTR, self).__init__()

        if backbone_name == 'resnet18s':  # Original LSTR backbone
            backbone = resnet18_reduced(
                pretrained=False, expansion=expansion,
                replace_stride_with_dilation=[False, False, False])
        else:  # Common backbones
            backbone = resnet.__dict__[backbone_name](
                pretrained=True,
                replace_stride_with_dilation=[False, True, True])

        return_layers = {'layer4': 'out'}
        self.backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)

        hidden_dim = 32 * expansion
        self.aux_loss = aux_loss
        self.position_embedding = build_position_encoding(hidden_dim=hidden_dim, position_embedding=pos_type)
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

        # Original LSTR: 3 classes + CE (softmax), we use binary classification with sigmoid to save parameters
        self.class_embed = nn.Linear(hidden_dim, 2)

        self.specific_embed = MLP(hidden_dim, hidden_dim, lsp_dim - 4, mlp_layers)  # Specific for each lane
        self.shared_embed = MLP(hidden_dim, hidden_dim, 4, mlp_layers)  # 4 shared curve coefficients

    def forward(self, images, padding_masks=None):
        # images: B x C x H x W
        # padding_masks: B x H x W (0 or 1 -> ignored)
        p = self.backbone(images)['out']

        # Padding mask (for paddings added in transforms)
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

    @torch.jit.unused
    def _set_aux_loss(self, output_class, output_curve):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{'logits': a, 'curves': b} for a, b in zip(output_class[:-1], output_curve[:-1])]
