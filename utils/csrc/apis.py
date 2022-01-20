from torch.utils.cpp_extension import load

line_nms = load(name='line_nms', sources=['line_nms/line_nms.cpp', 'line_nms/line_nms_kernel.cu'])
