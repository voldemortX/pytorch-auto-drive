import torch
from tools.tensorrt_utils import build_engine, do_inference

engine = build_engine("vgg16_baseline_tusimple_20210223.onnx")
dummy = torch.ones(1, 3, 360, 640)
context = engine.create_execution_context()
device = torch.device('cuda:0')
with torch.no_grad():
    in_t = dummy.contiguous().to(device)
    # trt_outputs = do_inference(engine=engine, input_tensor={'input1': dummy.contiguous().cuda()})
    trt_outputs = do_inference(engine=engine, input_tensor=in_t)

# print(trt_outputs)