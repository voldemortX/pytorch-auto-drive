import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit


def build_engine(model_path, max_batch_size=1, max_workspace_size=1 << 30):
    trt_logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(trt_logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    builder.max_batch_size = max_batch_size
    config = builder.create_builder_config()
    config.max_workspace_size = max_workspace_size
    parser = trt.OnnxParser(network, trt_logger)
    success = parser.parse_from_file(model_path)
    if success:
        print("correctly load onnx model...")
    else:
        for idx in range(parser.num_errors):
            print(parser.get_error(idx))
        raise ValueError
    serialized_engine = builder.build_serialized_network(network, config)

    return serialized_engine
