import torch
import time
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler
from torchvision_models.segmentation import erfnet_resnet
from data_processing import StandardLaneDetectionDataset, base_tusimple, base_culane
from transforms import ToTensor, Normalize, RandomHorizontalFlip, Resize, Compose
from all_utils_semseg import save_checkpoint


def erfnet_tusimple(num_classes, scnn=False, pretrained_weights='erfnet_encoder_pretrained.pth.tar'):
    # Define ERFNet for TuSimple (With only ImageNet pretraining)
    return erfnet_resnet(pretrained_weights=pretrained_weights, num_classes=num_classes, aux=6,
                         dropout_1=0.1, dropout_2=0.1, flattened_size=4500, scnn=scnn)


def erfnet_culane(num_classes, scnn=False, pretrained_weights='erfnet_encoder_pretrained.pth.tar'):
    # Define ERFNet for CULane (With only ImageNet pretraining)
    return erfnet_resnet(pretrained_weights=pretrained_weights, num_classes=num_classes, aux=4,
                         dropout_1=0.1, dropout_2=0.1, flattened_size=3965, scnn=scnn)


def init(batch_size, state, input_sizes, dataset, mean, std):
    # Return data_loaders
    # depending on whether the state is
    # 1: training
    # 2: just testing (validation set)
    # 3: just testing (test set)

    # Transformations
    # ! Can't use torchvision.Transforms.Compose
    if dataset == 'tusimple':
        base = base_tusimple
        workers = 4
    elif dataset == 'culane':
        base = base_culane
        workers = 4
    else:
        raise ValueError

    if state == 1:
        transforms = Compose(
            [ToTensor(),
             Resize(size_image=input_sizes[0], size_label=input_sizes[0]),
             RandomHorizontalFlip(flip_prob=0.5),
             Normalize(mean=mean, std=std)])
        data_set = StandardLaneDetectionDataset(root=base, image_set='train', transforms=transforms, data_set=dataset)
        data_loader = torch.utils.data.DataLoader(dataset=data_set, batch_size=batch_size,
                                                  num_workers=workers, shuffle=True)
    elif state == 2 or state == 3:
        transforms = Compose(
            [ToTensor(),
             Resize(size_image=input_sizes[0], size_label=input_sizes[0]),
             Normalize(mean=mean, std=std)])
        data_set = StandardLaneDetectionDataset(root=base, image_set='val' if state == 2 else 'test',
                                                transforms=transforms, data_set=dataset)
        data_loader = torch.utils.data.DataLoader(dataset=data_set, batch_size=batch_size,
                                                  num_workers=workers, shuffle=False)
    else:
        raise ValueError

    return data_loader


def train_schedule(writer, loader, save_num_steps, device, criterion, net, optimizer, lr_scheduler,
                   num_epochs, is_mixed_precision, exp_name):
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
                loss = criterion(inputs, labels, lane_existence, net)  # To support intermediate losses for SAD

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
            if current_step_num % save_num_steps == (save_num_steps - 1):
                save_checkpoint(net=net, optimizer=optimizer, lr_scheduler=lr_scheduler,
                                filename=exp_name + '_' + str(current_step_num) + '.pt')

        epoch += 1
        print('Epoch time: %.2fs' % (time.time() - time_now))


def test_one_set():
    # Evaluate on 1 data_loader
    pass
