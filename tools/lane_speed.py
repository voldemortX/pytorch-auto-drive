import torch
from tqdm import tqdm
from utils.datasets import StandardLaneDetectionDataset
from transforms import ToTensor, Normalize, Resize, Compose
import time


def init(input_sizes, dataset, mean, std, base, workers=0):
    transforms_test = Compose(
        [Resize(size_image=input_sizes, size_label=input_sizes),
         ToTensor(),
         Normalize(mean=mean, std=std)])

    validation_set = StandardLaneDetectionDataset(root=base, image_set='val', transforms=transforms_test,
                                                  data_set=dataset)
    validation_loader = torch.utils.data.DataLoader(dataset=validation_set, batch_size=1, num_workers=workers,
                                                    shuffle=False)
    return validation_loader


def lane_speed_evaluate(net, device, loader, num, count_interpolate=True):
    net.eval()
    iterable = iter(loader)

    # Warm-up hardware
    for _ in range(10):
        image, _ = iterable.__next__()
        image = image.to(device)
        _ = net(image)['out']

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
            output = net(image)['out']
            if count_interpolate:
                _ = torch.nn.functional.interpolate(output, size=image.shape[-2:], mode='bilinear', align_corners=True)
            torch.cuda.current_stream(device).synchronize()
            gpu_time += (time.perf_counter() - temp)

    fps = num / (io_time + gpu_time)
    gpu_fps = num / gpu_time

    return fps, gpu_fps


def lane_speed_evaluate_simple(net, device, dummy, num, count_interpolate=True):
    net.eval()
    dummy = dummy.to(device)
    output_size = dummy.shape[-2:]

    # Warm-up hardware
    for i in range(0, 10):
        _ = net(dummy)['out']

    # Timing
    torch.cuda.current_stream(device).synchronize()
    t_start = time.perf_counter()
    with torch.no_grad():
        for _ in tqdm(range(num)):
            output = net(dummy)['out']
            if count_interpolate:
                _ = torch.nn.functional.interpolate(output, size=output_size, mode='bilinear', align_corners=True)
    torch.cuda.current_stream(device).synchronize()
    fps_gpu = num / (time.perf_counter() - t_start)

    return fps_gpu
