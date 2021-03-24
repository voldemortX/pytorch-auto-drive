import yaml
import argparse
import torch
from torch.cuda.amp import autocast
from tqdm import tqdm
from utils.datasets import CULane, StandardLaneDetectionDataset
from transforms import ToTensor, Normalize, Resize, RandomRotation, Compose
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


def lane_speed_evaluate(net, device, loader, is_mixed_precision, output_size, num):
    count = 0
    net.eval()
    total_time = 0
    t1 = time.time()
    with torch.no_grad():
        for image, _ in tqdm(loader):
            if count == num:
                break
            image = image.to(device)
            t_start = time.time()
            output = net(image)['out']
            output = torch.nn.functional.interpolate(output, size=output_size, mode='bilinear', align_corners=True)
            t_end = time.time()
            total_time += (t_end - t_start)
            #print(total_time)
            count += 1
    fps = num / total_time
    t2 = time.time()
    print(t2 - t1)
    # print(total_time)
    # fps = 300 / (t2-t1)
    return fps


def lane_speed_evaluate_simple(net, device, is_mixed_precision, dummy, output_size, num):
    count = 0
    net.eval()
    total_time = 0
    dummy = dummy.to(device)
    t1 = time.time()
    #with torch.no_grad():
    for i in range(0, num):
        dummy = torch.randn((1, 3, 288, 800), device=device)
        t_start = time.time()
        output = net(dummy)['out']
        output = torch.nn.functional.interpolate(output, size=output_size, mode='bilinear', align_corners=True)
        t_end = time.time()
        total_time += (t_end - t_start)
        print(total_time)

    fps_wo_loader = num / total_time
    t2 = time.time()
    print(t2 - t1)

    return fps_wo_loader
