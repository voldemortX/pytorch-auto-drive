import torch
import tensorrt as trt


# TensorRT 7.2.3, context style
def build_engine(model_path, max_batch_size=1, max_workspace_size=1 << 30):
    engine_path = model_path[:model_path.rfind('.onnx')] + '.engine'
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    EXPLICIT_BATCH = max_batch_size << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    with trt.Builder(TRT_LOGGER) as builder, \
        builder.create_network(EXPLICIT_BATCH) as network, \
        trt.OnnxParser(network, TRT_LOGGER) as parser:
        with open(model_path, 'rb') as model:
            if not parser.parse(model.read()):
                for error in range(parser.num_errors):
                    print(parser.get_error(error))
            else:
                print("Correctly loaded ONNX model!")
        with trt.Builder(TRT_LOGGER) as builder, builder.create_builder_config() as config:
            config.max_workspace_size = max_workspace_size
            with builder.build_engine(network, config) as engine:
                serialized_engine = engine.serialize()
                save_engine(serialized_engine, engine_path)
                print('TensorRT engine saved at : {}'.format(engine_path))

    return engine_path

# def build_engine(model_path, max_batch_size=1, max_workspace_size=1 << 30):
#     trt_logger = trt.Logger(trt.Logger.WARNING)
#     builder = trt.Builder(trt_logger)
#     builder.max_batch_size = max_batch_size
#     config = builder.create_builder_config()
#     config.max_workspace_size = max_workspace_size
#     network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
#     parser = trt.OnnxParser(network, trt_logger)
#     success = parser.parse_from_file(model_path)
#     if success:
#         print("correctly load onnx model...")
#     else:
#         for idx in range(parser.num_errors):
#             print(parser.get_error(idx))
#         raise ValueError
#     serialized_engine = builder.build_serialized_network(network, config)
#     runtime = trt.Runtime(trt_logger)
#     engine = runtime.deserialize_cuda_engine(serialized_engine)
#
#     return engine


def save_engine(engine, engine_path):
    # Serialized engines are not portable across platforms or TensorRT versions.
    # Engines are specific to the exact GPU model they were built on
    with open(engine_path, "wb") as f:
        f.write(engine)


def load_engine(engine_path):
    # load the engine from a specific file
    with open(engine_path, "rb") as f:
        serialized_engine = f.read()

    return serialized_engine


def torch_device_from_trt(device):
    """Convert pytorch device to TensorRT device."""
    if device == trt.TensorLocation.DEVICE:
        return torch.device('cuda')
    elif device == trt.TensorLocation.HOST:
        return torch.device('cpu')
    else:
        return TypeError('%s is not supported by torch' % device)


def torch_dtype_from_trt(dtype):
    """Convert pytorch dtype to TensorRT dtype."""
    if dtype == trt.bool:
        return torch.bool
    elif dtype == trt.int8:
        return torch.int8
    elif dtype == trt.int32:
        return torch.int32
    elif dtype == trt.float16:
        return torch.float16
    elif dtype == trt.float32:
        return torch.float32
    else:
        raise TypeError('%s is not supported by torch' % dtype)


def inference_trt(engine_binary, input_tensor):
    # Because the engine is converted from onnx model,
    # the input_names and output_names should be the same as onnx model.
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(TRT_LOGGER)
    engine = runtime.deserialize_cuda_engine(engine_binary)
    context = engine.create_execution_context()

    # Inference
    names = [_ for _ in engine]
    input_names = list(filter(engine.binding_is_input, names))
    output_names = list(set(names) - set(input_names))
    num_bindings = len(input_names) + len(output_names)
    bindings = [None] * num_bindings

    for name in input_names:
        input_idx = engine.get_binding_index(name)
        context.set_binding_shape(input_idx, tuple(input_tensor.shape))
        bindings[input_idx] = input_tensor.contiguous().data_ptr()

    outputs = {}
    for name in output_names:
        output_idx = engine.get_binding_index(name)
        dtype = torch_dtype_from_trt(engine.get_binding_dtype(output_idx))
        shape = tuple(context.get_binding_shape(output_idx))
        device = torch_device_from_trt(engine.get_location(output_idx))
        output = torch.empty(size=shape, dtype=dtype, device=device)
        outputs[name] = output
        bindings[output_idx] = output.data_ptr()

    context.execute_async_v2(bindings, torch.cuda.current_stream().cuda_stream)

    return outputs
