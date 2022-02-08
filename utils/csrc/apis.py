import os
from torch.utils.cpp_extension import load

csrc_path = 'utils/csrc'

line_nms_ = load(name='line_nms',
                 sources=[os.path.join(csrc_path, 'line_nms', 'line_nms.cpp'),
                          os.path.join(csrc_path, 'line_nms', 'line_nms_kernel.cu')],
                 verbose=False)


# Wrap it to be like a normal Python func
# TODO: cpu version
def line_nms(boxes, scores, overlap, top_k):
    # Notes: removed the extra 5 (start, end, len, valid (2)) in original LaneATT

    return line_nms_.forward(boxes.contiguous(), scores.contiguous(), overlap, top_k)
