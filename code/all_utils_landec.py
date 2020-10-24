import os
import cv2
import torch
import time
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler
from torchvision_models.segmentation import erfnet_resnet
from data_processing import StandardLaneDetectionDataset, base_tusimple, base_culane
from transforms import ToTensor, Normalize, Resize, RandomRotation, Compose
from all_utils_semseg import save_checkpoint


def erfnet_tusimple(num_classes, scnn=False, pretrained_weights='erfnet_encoder_pretrained.pth.tar'):
    # Define ERFNet for TuSimple (With only ImageNet pretraining)
    return erfnet_resnet(pretrained_weights=pretrained_weights, num_classes=num_classes, aux=6,
                         dropout_1=0.1, dropout_2=0.1, flattened_size=2560, scnn=scnn)


def erfnet_culane(num_classes, scnn=False, pretrained_weights='erfnet_encoder_pretrained.pth.tar'):
    # Define ERFNet for CULane (With only ImageNet pretraining)
    return erfnet_resnet(pretrained_weights=pretrained_weights, num_classes=num_classes, aux=4,
                         dropout_1=0.1, dropout_2=0.1, flattened_size=4500, scnn=scnn)


def init(batch_size, state, input_sizes, dataset, mean, std):
    # Return data_loaders
    # depending on whether the state is
    # 0: training
    # 1: just testing (validation set)
    # 2: just testing (test set)

    # Transformations
    # ! Can't use torchvision.Transforms.Compose
    if dataset == 'tusimple':
        base = base_tusimple
        workers = 8
    elif dataset == 'culane':
        base = base_culane
        workers = 10
    else:
        raise ValueError

    if state == 0:
        transforms = Compose(
            [RandomRotation(degrees=1),
             ToTensor(),
             Resize(size_image=input_sizes[0], size_label=input_sizes[0]),
             Normalize(mean=mean, std=std)])
        data_set = StandardLaneDetectionDataset(root=base, image_set='train', transforms=transforms, data_set=dataset)
        data_loader = torch.utils.data.DataLoader(dataset=data_set, batch_size=batch_size,
                                                  num_workers=workers, shuffle=True)
    elif state == 1 or state == 2:
        transforms = Compose(
            [ToTensor(),
             Resize(size_image=input_sizes[0], size_label=input_sizes[0]),
             Normalize(mean=mean, std=std)])
        data_set = StandardLaneDetectionDataset(root=base, image_set='val' if state == 1 else 'test',
                                                transforms=transforms, data_set=dataset)
        data_loader = torch.utils.data.DataLoader(dataset=data_set, batch_size=batch_size,
                                                  num_workers=workers, shuffle=False)
    else:
        raise ValueError

    return data_loader


def train_schedule(writer, loader, save_num_steps, device, criterion, net, optimizer, lr_scheduler,
                   num_epochs, is_mixed_precision, input_sizes, exp_name):
    # Should be the same as segmentation, given customized loss classes
    net.train()
    epoch = 0
    loss_num_steps = int(len(loader) / 10) if len(loader) > 10 else 1
    if is_mixed_precision:
        scaler = GradScaler()

    # Training
    while epoch < num_epochs:
        net.train()
        running_loss = 0.0
        time_now = time.time()
        for i, data in enumerate(loader, 0):
            inputs, labels, lane_existence = data
            inputs, labels, lane_existence = inputs.to(device), labels.to(device), lane_existence.to(device)
            optimizer.zero_grad()

            with autocast(is_mixed_precision):
                # To support intermediate losses for SAD
                loss = criterion(inputs, labels, lane_existence, net, input_sizes[0])

            if is_mixed_precision:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            lr_scheduler.step()
            running_loss += loss.item()
            current_step_num = int(epoch * len(loader) + i + 1)

            # Record losses
            if current_step_num % loss_num_steps == (loss_num_steps - 1):
                print('[%d, %d] loss: %.4f' % (epoch + 1, i + 1, running_loss / 100))
                writer.add_scalar('training loss',
                                  running_loss / 100,
                                  current_step_num)
                running_loss = 0.0

            # Record checkpoints
            if current_step_num % save_num_steps == (save_num_steps - 1) or \
               current_step_num == num_epochs * len(loader):
                save_checkpoint(net=net, optimizer=optimizer, lr_scheduler=lr_scheduler,
                                filename=exp_name + '_' + str(current_step_num) + '.pt')

        epoch += 1
        print('Epoch time: %.2fs' % (time.time() - time_now))


# Adapted from harryhan618/SCNN_Pytorch
def test_one_set(net, device, loader, is_mixed_precision, input_sizes, gap, ppl):
    # Predict on 1 data_loader and save predictions for the official script

    net.eval()
    with torch.no_grad():
        for images, filenames in tqdm(loader):
            images = images.to(device)

            with autocast(is_mixed_precision):
                outputs = net(images)
                prob_map = torch.nn.functional.interpolate(outputs['out'], size=input_sizes[0], mode='bilinear',
                                                           align_corners=True).softmax(dim=1)
                existence = (outputs['aux'] > 0.5)

            # To CPU
            prob_map = prob_map.cpu().numpy()
            existence = existence.cpu().numpy()

            # Get coordinates for lanes
            for j in range(existence.shape[0]):
                lane_coordinates = prob_to_lines(prob_map[j], existence[j], resize_shape=input_sizes[1],
                                                 gap=gap, ppl=ppl)

                # Save lanes to disk
                dir_name = filenames[j][:filenames[j].rfind('/')]
                if not os.path.exists(dir_name):
                    os.makedirs(dir_name)
                with open(filenames[j], "w") as f:
                    for lane in lane_coordinates:
                        if lane:  # No printing for []
                            for (x, y) in lane:
                                print("{} {}".format(x, y), end=" ", file=f)
                            print(file=f)


# Adapted from harryhan618/SCNN_Pytorch
def get_lane(prob_map, gap, ppl, thresh, resize_shape=None):
    """
    Arguments:
    ----------
    prob_map: prob map for single lane, np array size (h, w)
    resize_shape:  reshape size target, (H, W)
    Return:
    ----------
    coords: x coords bottom up every gap px, 0 for non-exist, in resized shape
    """

    if resize_shape is None:
        resize_shape = prob_map.shape
    h, w = prob_map.shape
    H, W = resize_shape

    coords = np.zeros(ppl)
    for i in range(ppl):
        y = int(h - i * gap / H * h - 1)
        if y < 0:
            break
        line = prob_map[y, :]
        id = np.argmax(line)
        if line[id] > thresh:
            coords[i] = int(id / w * W)
    if (coords > 0).sum() < 2:
        coords = np.zeros(ppl)
    return coords


# Adapted from harryhan618/SCNN_Pytorch
def prob_to_lines(seg_pred, exist, resize_shape=None, smooth=True, gap=20, ppl=None, thresh=0.3):
    """
    Arguments:
    ----------
    seg_pred: np.array size (num_classes, h, w)
    resize_shape:  reshape size target, (H, W)
    exist:   list of existence, e.g. [0, 1, 1, 0]
    smooth:  whether to smooth the probability or not
    gap: y pixel gap for sampling
    ppl:     how many points for one lane
    thresh:  probability threshold
    Return:
    ----------
    coordinates: [x, y] list of lanes, e.g.: [ [[9, 569], [50, 549]] ,[[630, 569], [647, 549]] ]
    """
    if resize_shape is None:
        resize_shape = seg_pred.shape[1:]  # seg_pred (num_classes, h, w)
    _, h, w = seg_pred.shape
    H, W = resize_shape
    coordinates = []

    if ppl is None:
        ppl = round(H / 2 / gap)

    for i in range(1, seg_pred.shape[0]):
        prob_map = seg_pred[i, :, :]
        if smooth:
            prob_map = cv2.blur(prob_map, (9, 9), borderType=cv2.BORDER_REPLICATE)
        if exist[i - 1]:
            coords = get_lane(prob_map, gap, ppl, thresh, resize_shape)
            coordinates.append([[coords[j], H - 1 - j * gap] for j in range(ppl) if coords[j] > 0])

    return coordinates
