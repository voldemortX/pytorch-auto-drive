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
                                                    shuffle=True)
    return validation_loader


def lane_speed_evaluate(net, device, loader, is_mixed_precision, output_size):
    count = 0
    net.eval()
    total_time = 0
    with torch.no_grad():
        for image, _ in tqdm(loader):
            if count == 300:
                break
            image = image.to(device)
            t_start = time.time()
            with autocast(is_mixed_precision):
                output = net(image)['out']
                output = torch.nn.functional.interpolate(output, size=output_size, mode='bilinear', align_corners=True)
            t_end = time.time()
            total_time += (t_end - t_start)
            count += 1

    fps = 300 / total_time
    print(total_time)
    return fps
