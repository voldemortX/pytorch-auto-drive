from tools.tensorrt_utils import build_engine

engine = build_engine("vgg16_baseline_tusimple_20210223.onnx")
print(engine)