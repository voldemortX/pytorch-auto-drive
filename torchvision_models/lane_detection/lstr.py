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
                 flag=False,
                 block=None,
                 layers=None,
                 res_dims=None,
                 res_strides=None,
                 attn_dim=None,
                 num_queries=None,
                 aux_loss=None,
                 pos_type=None,
                 drop_out=0.1,
                 num_heads=None,
                 dim_feedforward=None,
                 enc_layers=None,
                 dec_layers=None,
                 pre_norm=None,
                 return_intermediate=None,
                 lsp_dim=None,
                 mlp_layers=None,
                 num_cls=None,
                 ):
        super(LSTR, self).__init__()
        self.flag = flag

        backbone = resnet18_reduced(
            pretrained=False,
            replace_stride_with_dilation=[False, False, False])
        return_layers = {'layer4': 'out'}
        self.backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)

        hidden_dim = attn_dim
        self.aux_loss = aux_loss
        self.position_embedding = build_position_encoding(hidden_dim=hidden_dim, position_embedding=pos_type)
        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        self.input_proj = nn.Conv2d(res_dims[-1], hidden_dim, kernel_size=1)  # the same as channel of self.layer4

        self.transformer = build_transformer(hidden_dim=hidden_dim,
                                             dropout=drop_out,
                                             nheads=num_heads,
                                             dim_feedforward=dim_feedforward,
                                             enc_layers=enc_layers,
                                             dec_layers=dec_layers,
                                             pre_norm=pre_norm,
                                             return_intermediate_dec=return_intermediate)

        self.class_embed = nn.Linear(hidden_dim, num_cls + 1)
        self.specific_embed = MLP(hidden_dim, hidden_dim, lsp_dim - 4, mlp_layers)
        self.shared_embed = MLP(hidden_dim, hidden_dim, 4, mlp_layers)

    def _train(self, images, interp_size):
        p = self.backbone(images)['out']
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

    def _test(self, images, interp_size):
        return self._train(images, interp_size)

    def forward(self, images, interp_size):
        if self.flag:
            return self._train(images, interp_size)
        else:
            return self._test(images, interp_size)

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{'logits': a, 'curves': b} for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]
