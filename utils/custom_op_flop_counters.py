import torch


def hook_DCN_v2_Ref(m, x, y):
    # Only modulated_deform_conv2d() is ignored in feature flip fusion,
    # same as default DCNv2.
    # conv Flops = 2 x k^2 x Cin x Cout x W x H
    # dcn Flops = + W x H x k^2 x 13 + W x H x k^2 x 9 x Cin (bilinear interpolate, 9 -> 6 for MACs)
    # or just: ->0 + W x H x k^2 x 9 x Cin (3 weighted plus, 9 -> 6 for MACs)
    # ->0 coordinates: W x H x k^2 x 13
    # DCNv2: ++ W x H x k^2 x Cin
    # ~ (additional flops on bias is not considered):
    x = x[0]
    C_in, fH, fW = x.shape[1:]
    C_out = y.shape[1]

    # 1. Offset conv
    bias_ops = 1 if m.conv_offset.bias is not None else 0

    # N x Cout x H x W x  (Cin x Kw x Kh + bias)
    m.total_ops += counter_conv(
        bias_ops,
        torch.zeros(m.conv_offset.weight.size()[2:]).numel(),
        int(m.conv_offset.weight.size()[0] * fH * fW),
        m.conv_offset.in_channels,
        m.conv_offset.groups,
    )

    # 2. sigmoid on DCNv2 mask (doubly count as in thop)
    m.total_ops += torch.DoubleTensor([19 * m.conv_offset.weight.size()[1] * fH * fW // 3])

    # 3. DCNv2
    m.total_ops += torch.DoubleTensor([count_one_dcn_v2(fH, fW, C_in, C_out)])


def counter_conv(bias, kernel_size, output_size, in_channel, group):
    """inputs are all numbers!"""
    return torch.DoubleTensor([output_size * (in_channel / group * kernel_size + bias)])


def count_one_dcn_v2(fW, fH, C_in, C_out):
    standard_conv_macs = fW * fH * (3 * 3) * C_in * C_out
    ignored_dcnv2_macs = 7 * fW * fH * (3 * 3) * C_in + fW * fH * (3 * 3) * 13

    return standard_conv_macs + ignored_dcnv2_macs
