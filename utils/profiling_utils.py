import torch
import time
from tqdm import tqdm
from thop import profile

from .datasets import DATASETS
from .transforms import TRANSFORMS
from .custom_op_flop_counters import hook_DCN_v2_Ref
from .models.common_models import DCN_v2_Ref

REPLACE_LIST = [
    'Resize',
    'ZeroPad'
]

REPLACE_KEY_LIST = [
    'image_size',
    'size'
]


def transforms_surgeon(transforms_cfg, class_to_replace, key_to_replace, value):
    for i in range(len(transforms_cfg['transforms'])):
        if transforms_cfg['transforms'][len(transforms_cfg['transforms']) - i - 1]['name'] == class_to_replace:
            transforms_cfg['transforms'][len(transforms_cfg['transforms']) - i - 1][key_to_replace] = value
            return True, transforms_cfg

    return False, transforms_cfg


def init_dataset(dataset_cfg, transforms_cfg, input_sizes):
    # Find last resize
    flag = False
    for ctr, ktr in zip(REPLACE_LIST, REPLACE_KEY_LIST):
        flag, transforms_cfg = transforms_surgeon(transforms_cfg, ctr, ktr, value=input_sizes)
        if flag:
            break
    assert flag is True, 'Can\'t find transform to replace size, check your `test_augmentation`!'

    # Dataset
    dataset = DATASETS.from_dict(dataset_cfg, transforms=TRANSFORMS.from_dict(transforms_cfg))
    val_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=1, num_workers=0, shuffle=False)

    return val_loader


def speed_evaluate_real(net, device, loader, num):
    net.eval(profiling=True)
    iterable = iter(loader)

    # Warm-up hardware
    with torch.no_grad():
        for _ in range(10):
            image, _ = iterable.__next__()
            image = image.to(device)
            _ = net(image)

    # Timing with loading images from disk
    gpu_time = 0
    io_time = 0
    with torch.no_grad():
        for _ in tqdm(range(num)):
            # I/O
            torch.cuda.current_stream(device).synchronize()
            temp = time.perf_counter()
            image, _ = iterable.__next__()
            image = image.to(device)
            torch.cuda.current_stream(device).synchronize()
            io_time += (time.perf_counter() - temp)

            # GPU
            torch.cuda.current_stream(device).synchronize()
            temp = time.perf_counter()
            output = net(image)
            if 'out' in output.keys() and output['out'] != image.shape:
                _ = torch.nn.functional.interpolate(output['out'], size=image.shape[-2:],
                                                    mode='bilinear', align_corners=True)
            torch.cuda.current_stream(device).synchronize()
            gpu_time += (time.perf_counter() - temp)

    fps = num / (io_time + gpu_time)
    gpu_fps = num / gpu_time

    return fps, gpu_fps


def speed_evaluate_simple(net, device, dummy, num):
    net.eval(profiling=True)
    dummy = dummy.to(device)
    output_size = dummy.shape[-2:]

    # Warm-up hardware
    with torch.no_grad():
        for i in range(0, 10):
            _ = net(dummy)

    # Timing
    torch.cuda.current_stream(device).synchronize()
    t_start = time.perf_counter()
    with torch.no_grad():
        for _ in tqdm(range(num)):
            output = net(dummy)
            if 'out' in output.keys() and output['out'] != dummy.shape:
                _ = torch.nn.functional.interpolate(output['out'], size=output_size,
                                                    mode='bilinear', align_corners=True)
    torch.cuda.current_stream(device).synchronize()
    fps_gpu = num / (time.perf_counter() - t_start)

    return fps_gpu


def model_profile(net, height, width, device):
    temp = torch.randn(1, 3, height, width).to(device)
    macs, params = profile(net, inputs=(temp,),
                           custom_ops={DCN_v2_Ref: hook_DCN_v2_Ref})

    return macs, params
