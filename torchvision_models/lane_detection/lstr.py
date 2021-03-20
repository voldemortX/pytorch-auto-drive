# Adapted from liuruijin17/LSTR
# TODO: Why no dilations and use frozen BN on first BN layer? Check implementation details of ResNet18
import torch
import torch.nn as nn
import torch.nn.functional as F
from ..transformer import build_transformer, build_position_encoding
from ..resnet import resnet18_reduced
from ..mlp import MLP
from .._utils import IntermediateLayerGetter


class LSTR(nn.Module):
    def __init__(self,
                 expansion=1,  # Expansion rate (1x for TuSimple & 2x for CULane)
                 num_queries=7,
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
                 num_cls=2,  # Lane or Not
                 ):
        super(LSTR, self).__init__()

        backbone = resnet18_reduced(
            pretrained=False, expansion=expansion,
            replace_stride_with_dilation=[False, False, False])
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

        self.class_embed = nn.Linear(hidden_dim, num_cls + 1)
        self.specific_embed = MLP(hidden_dim, hidden_dim, lsp_dim - 4, mlp_layers)
        self.shared_embed = MLP(hidden_dim, hidden_dim, 4, mlp_layers)

    def forward(self, images, interp_size):
        p = self.backbone(images)['out']

        # Padding mask (transformer need fix sized input)
        # TODO: what is this mask in LSTR?
        pmasks = F.interpolate(interp_size, size=p.shape[-2:]).to(torch.bool)[0]

        pos = self.position_embedding(p, pmasks)
        hs, _ = self.transformer(self.input_proj(p), pmasks, self.query_embed.weight, pos)
        output_class = self.class_embed(hs)
        output_specific = self.specific_embed(hs)
        output_shared = self.shared_embed(hs)
        output_shared = torch.mean(output_shared, dim=-2, keepdim=True)
        output_shared = output_shared.repeat(1, 1, output_specific.shape[2], 1)
        output_specific = torch.cat([output_specific[:, :, :, :2],
                                     output_shared, output_specific[:, :, :, 2:]], dim=-1)
        out = {'logits': output_class[-1], 'curves': output_specific[-1]}
        if self.aux_loss:
            out['lane'] = self._set_aux_loss(output_class, output_specific)

        return out

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{'logits': a, 'curves': b} for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]
