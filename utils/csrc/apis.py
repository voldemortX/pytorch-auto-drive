import os
from torch.utils.cpp_extension import load

csrc_path = 'utils/csrc'

line_nms = load(name='line_nms', sources=[os.path.join(csrc_path, 'line_nms', 'line_nms.cpp'),
                                          os.path.join(csrc_path, 'line_nms', 'line_nms_kernel.cu')])
