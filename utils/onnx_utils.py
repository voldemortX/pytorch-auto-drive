# Convert only the pt model part

import onnx
import onnxruntime as ort
import numpy as np
import torch


DEFAULT_OPSET_VERSION = 9
MINIMAL_OPSET_VERSIONS = {
    # Others use 9
    'LSTR': 10,
    'RESA': 11,
    'SpatialConv': 11
}
TRACE_REQUIRE_PREPROCESSING = [
    'LSTR',
    'RESA'
]


def get_minimal_opset_version(cfg, min_version):
    # Recursively get minimum version
    if isinstance(cfg, dict):
        for k, v in cfg.items():
            if k == 'name':
                temp = MINIMAL_OPSET_VERSIONS.get(v)
                if temp is None:
                    temp = DEFAULT_OPSET_VERSION
                min_version = max(min_version, temp)
            else:
                min_version = max(min_version, get_minimal_opset_version(v, min_version))
        return min_version
    else:
        return DEFAULT_OPSET_VERSION


def append_trace_arg(cfg, trace_arg):
    # Do the above trick again
    if isinstance(cfg, dict) and cfg.get('name') in TRACE_REQUIRE_PREPROCESSING:
        cfg['trace_arg'] = trace_arg
    else:
        for k in cfg.keys():
            cfg[k] = append_trace_arg(cfg[k], trace_arg)

    return cfg


def pt_to_onnx(net, dummy, filename, opset_version=9):
    net.eval()
    temp = net(dummy)
    torch.onnx.export(net, dummy, filename, verbose=True, input_names=['input1'], output_names=temp.keys(),
                      opset_version=opset_version)


@torch.no_grad()
def test_conversion(pt_net, onnx_filename, dummy):
    pt_net.eval()
    pt_out = pt_net(dummy)
    dummy = dummy.cpu()
    onnx_out = inference_onnx(dummy, onnx_filename)
    diff = 0.0
    avg = 0.0
    for k, temp_pt in pt_out.items():
        temp_onnx = onnx_out[k]
        diff += np.abs((temp_onnx - temp_pt.cpu().numpy())).mean()
        avg += temp_pt.abs().mean().item()
    diff /= len(onnx_out)
    avg /= len(onnx_out)
    diff_percentage = diff / avg * 100
    print('Average diff: {}\nAverage diff (%): {}'.format(diff, diff_percentage))
    assert diff_percentage < 0.1, 'Diff over 0.1%, please check for special operators!'


def inference_onnx(dummy, onnx_filename):
    onnx_net = onnx.load(onnx_filename)
    onnx.checker.check_model(onnx_net)
    onnx.helper.printable_graph(onnx_net.graph)
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    ort_session = ort.InferenceSession(onnx_filename, providers=providers)
    print(ort_session.get_providers())
    input_all = [node.name for node in onnx_net.graph.input]
    input_initializer = [node.name for node in onnx_net.graph.initializer]
    input_names = list(set(input_all) - set(input_initializer))
    onnx_out = ort_session.run(None, {input_names[0]: dummy.numpy()})
    output_names = [node.name for node in onnx_net.graph.output]

    return {k: v for k, v in zip(output_names, onnx_out)}
