import torch
import time
from tqdm import tqdm
from utils.datasets import StandardLaneDetectionDataset
from transforms import ToTensor, Normalize, Resize, Compose, ZeroPad, LabelMap
from utils.datasets import StandardSegmentationDataset
from thop import profile


def init_lane(input_sizes, dataset, mean, std, base, workers=0):
    transforms_test = Compose(
        [Resize(size_image=input_sizes, size_label=input_sizes),
         ToTensor(),
         Normalize(mean=mean, std=std)])
    validation_set = StandardLaneDetectionDataset(root=base, image_set='val', transforms=transforms_test,
                                                  data_set=dataset)
    validation_loader = torch.utils.data.DataLoader(dataset=validation_set, batch_size=1, num_workers=workers,
                                                    shuffle=False)
    return validation_loader


def init_seg(input_sizes, std, mean, dataset, test_base=None, test_label_id_map=None, city_aug=0):
    if dataset == 'voc':
        transform_test = Compose(
            [ToTensor(),
             ZeroPad(size=input_sizes),
             Normalize(mean=mean, std=std)])
    elif dataset == 'city' or dataset == 'gtav' or dataset == 'synthia':  # All the same size
        if city_aug == 2:  # ERFNet and ENet
            transform_test = Compose(
                [ToTensor(),
                 Resize(size_image=input_sizes, size_label=input_sizes),
                 LabelMap(test_label_id_map)])
        elif city_aug == 1:  # City big
            transform_test = Compose(
                [ToTensor(),
                 Resize(size_image=input_sizes, size_label=input_sizes),
                 Normalize(mean=mean, std=std),
                 LabelMap(test_label_id_map)])
    else:
        raise ValueError

    # Not the actual test set (i.e. validation set)
    test_set = StandardSegmentationDataset(root=test_base, image_set='val', transforms=transform_test,
                                           data_set='city' if dataset == 'gtav' or dataset == 'synthia' else dataset)

    val_loader = torch.utils.data.DataLoader(dataset=test_set, batch_size=1, num_workers=0,
                                             shuffle=False)

    # Testing
    return val_loader


def speed_evaluate_real(net, device, loader, num, count_interpolate=True):
    net.eval()
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
            if count_interpolate:
                _ = torch.nn.functional.interpolate(output['out'], size=image.shape[-2:],
                                                    mode='bilinear', align_corners=True)
            torch.cuda.current_stream(device).synchronize()
            gpu_time += (time.perf_counter() - temp)

    fps = num / (io_time + gpu_time)
    gpu_fps = num / gpu_time

    return fps, gpu_fps


def speed_evaluate_simple(net, device, dummy, num, count_interpolate=True):
    net.eval()
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
            if count_interpolate:
                _ = torch.nn.functional.interpolate(output['out'], size=output_size,
                                                    mode='bilinear', align_corners=True)
    torch.cuda.current_stream(device).synchronize()
    fps_gpu = num / (time.perf_counter() - t_start)

    return fps_gpu


def model_profile(net, height, width, device):
    temp = torch.randn(1, 3, height, width).to(device)
    macs, params = profile(net, inputs=(temp,))

    return macs, params

