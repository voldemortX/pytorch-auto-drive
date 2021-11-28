# Convert only the pt model part

import onnx
import onnxruntime as ort
import numpy as np
import torch

MINIMAL_OPSET_VERSIONS = {
    # Others use 9
    'lstr': 11,
    'resa': 12,
    'scnn': 11
}


def add_basic_arguments(p):
    p.add_argument('--height', type=int, default=288,
                   help='Image input height (default: 288)')
    p.add_argument('--width', type=int, default=800,
                   help='Image input width (default: 800)')
    p.add_argument('--dataset', type=str, default='tusimple',
                   help='Profile on TuSimple (tusimple) / CULane (culane) (default: tusimple)')
    p.add_argument('--method', type=str, default='baseline',
                   help='method selection (lstr/scnn/sad/baseline) (default: baseline)')
    p.add_argument('--backbone', type=str, default='erfnet',
                   help='backbone selection (erfnet/enet/vgg16/resnet18s/resnet18/resnet34/resnet50/resnet101)'
                        '(default: erfnet)')
    p.add_argument('--task', type=str, default='lane',
                   help='task selection (lane/seg)')
    p.add_argument('--model', type=str, default='deeplabv3',
                   help='Model selection (fcn/erfnet/deeplabv2/deeplabv3/enet) (default: deeplabv3)')
    p.add_argument('--encoder-only', action='store_true', default=False,
                   help='Only train the encoder. ENet trains encoder and decoder separately (default: False)')
    p.add_argument('--continue-from', type=str, default=None,
                   help='Continue training from a previous checkpoint')
    p.add_argument('--batch-size', type=int, default=8,
                   help='input batch size. Recommend 4 times the training batch size in testing (default: 8)')
    p.add_argument('--mixed-precision', action='store_true', default=False,
                   help='Enable mixed precision training (default: False)')
    p.add_argument('--state', type=int, default=0,
                   help='Conduct validation(3)/final test(2)/fast validation(1)/normal training(0) (default: 0)')
    p.add_argument('--workers', type=int, default=10,
                   help='Number of workers (threads) when loading data.'
                        'Recommend value for training: batch_size / 2 (default: 10)')
    p.add_argument('--exp-name', type=str, default='',
                   help='Name of experiment')
    p.add_argument('--verify', type=str, default='real',
                   help='no: without verification/real: process the whole dataset/simple: process a random tensor')


def pt_to_onnx(net, dummy, filename, opset_version=9):
    net.eval()
    torch.onnx.export(net, dummy, filename, verbose=True, input_names=['input1'], output_names=['output1'],
                      opset_version=opset_version)


@torch.no_grad()
def test_conversion(pt_net, onnx_filename, dummy):
    pt_net.eval()
    dummy = dummy.cpu()
    pt_net = pt_net.cpu()
    pt_out = pt_net(dummy)
    onnx_net = onnx.load(onnx_filename)
    onnx.checker.check_model(onnx_net)
    onnx.helper.printable_graph(onnx_net.graph)
    ort_session = ort.InferenceSession(onnx_filename)
    onnx_out = ort_session.run(None, {'input1': dummy.numpy()})
    diff = 0.0
    avg = 0.0
    for (_, temp_pt), temp_onnx in zip(pt_out.items(), onnx_out):
        diff += np.abs((temp_onnx - temp_pt.numpy())).mean()
        avg += temp_pt.abs().mean().item()
    diff /= len(onnx_out)
    avg /= len(onnx_out)
    diff_percentage = diff / avg * 100
    print('Average diff: {}\nAverage diff (%): {}'.format(diff, diff_percentage))
    assert diff_percentage < 0.1, 'Diff over 0.1%, please check for special operators!'


def get_ort_session(onnx_filename):
    # return onnx runtime session
    print(ort.get_device())
    # providers = [
    #     ('CUDAExecutionProvider', {
    #         'device_id': 0,
    #         'arena_extend_strategy': 'kNextPowerOfTwo',
    #         'gpu_mem_limit': 2 * 1024 * 1024 * 1024,
    #         'cudnn_conv_algo_search': 'EXHAUSTIVE',
    #         'do_copy_in_default_stream': True,
    #     }),
    # ]
    onnx_net = onnx.load(onnx_filename)
    onnx.checker.check_model(onnx_net)
    onnx.helper.printable_graph(onnx_net.graph)
    ort_session = ort.InferenceSession(onnx_filename)

    return ort_session


def to_numpy(tensor):
    # transfer tensor to numpy
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()
