from importmagician import import_from
with import_from('./'):
    from utils.models.common_models import DCN_v2_Ref

import torch
# from fvcore.nn import FlopCountAnalysis
from thop import profile


def thop_count(x, y, f):
    macs, params = profile(f, inputs=(x, y))

    return macs * 2, params


# def fvcore_count(x, y, f):
#     flops = FlopCountAnalysis(f, (x, y))

#     return flops.total() * 2, 0


H = 360
W = 640
C = 256
OS = 16
fH = (H - 1) // OS + 1
fW = (W - 1) // OS + 1

inputs1 = torch.ones(1, C, fH, fW).cuda()
inputs2 = torch.ones(1, C, (H - 1) // OS + 1, (W - 1) // OS + 1).cuda()
# model = SimpleFlip_2D(channels=256).cuda()
model = DCN_v2_Ref(C, C, kernel_size=(3, 3), padding=1).cuda()
model.eval()
print(thop_count(inputs1, inputs2, model))  # Also doubly counts sigmoid (damn the activations)
# print(fvcore_count(inputs1, inputs2, model))


# Only modulated_deform_conv2d() is ignored
# conv Flops = 2 x k^2 x Cin x Cout x W x H
# dcn Flops = + W x H x k^2 x 13 + W x H x k^2 x 9 x Cin (bilinear interpolate)
# or just: ->0 + W x H x k^2 x 9 x Cin (3 weighted plus)
# ->0 coordinates: W x H x k^2 x 13
# dcnv2: ++ W x H x k^2 x Cin
# ~ (additional flops on bias is not considered):
standard_conv_flops = fW * fH * (3 * 3) * C * C
ignored_dcnv2_flops = 10 * fW * fH * (3 * 3) * C + fW * fH * (3 * 3) * 13
print('DCN_v2_Ref ignored flops by thop:')
print(standard_conv_flops + ignored_dcnv2_flops)
