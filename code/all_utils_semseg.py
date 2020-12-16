import torch
import time
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler

from torchvision_models.segmentation import deeplabv2_resnet101, deeplabv3_resnet101, fcn_resnet101, erfnet_resnet
from data_processing import StandardSegmentationDataset, base_city, base_voc, base_gtav, base_synthia, \
                            label_id_map_city, label_id_map_synthia, iou_13, iou_16
from transforms import ToTensor, Normalize, RandomHorizontalFlip, Resize, RandomResize, RandomCrop, RandomTranslation,\
                       ZeroPad, LabelMap, RandomScale, Compose


def fcn(num_classes):
    # Define FCN with ResNet101 (With only ImageNet pretraining)
    return fcn_resnet101(pretrained=False, num_classes=num_classes)


def deeplab_v3(num_classes):
    # Define deeplabV3 with ResNet101 (With only ImageNet pretraining)
    return deeplabv3_resnet101(pretrained=False, num_classes=num_classes)


def deeplab_v2(num_classes):
    # Define deeplabV2 with ResNet101 (With only ImageNet pretraining)
    return deeplabv2_resnet101(pretrained=False, num_classes=num_classes)


def erfnet(num_classes, pretrained_weights='erfnet_encoder_pretrained.pth.tar'):
    # Define ERFNet (With only ImageNet pretraining)
    return erfnet_resnet(pretrained_weights=pretrained_weights, num_classes=num_classes)


# Copied and simplified from torch/vision/references/segmentation
class ConfusionMatrix(object):
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.mat = None

    def update(self, a, b):
        n = self.num_classes
        if self.mat is None:
            self.mat = torch.zeros((n, n), dtype=torch.int64, device=a.device)
        with torch.no_grad():
            k = (a >= 0) & (a < n)
            inds = n * a[k].to(torch.int64) + b[k]
            self.mat += torch.bincount(inds, minlength=n**2).reshape(n, n)

    def reset(self):
        self.mat.zero_()

    def compute(self):
        h = self.mat.float()
        acc_global = torch.diag(h).sum() / h.sum()
        acc = torch.diag(h) / h.sum(1)
        iu = torch.diag(h) / (h.sum(1) + h.sum(0) - torch.diag(h))
        return acc_global, acc, iu


def show(images, is_label, colors, std, mean):
    # Draw images/labels from tensors
    np_images = images.numpy()
    if is_label:
        # Map to RGB((N, d1, d2) = {0~20, 255} => (N, d1, d2, 3) = {0.0~1.0})
        # As for how I managed this, I literally have no clue,
        # but it seems to be working
        np_images = np_images.reshape((np_images.shape[0], np_images.shape[1], np_images.shape[2], 1))
        np_images = np.tile(np_images, (1, 1, 1, 3))
        np_images[np_images == 255] = 21  # Ignore 255
        np_images = np.array(colors)[np_images[:, :, :, 0]]
        np_images = np_images / 255.0
    else:
        # Denormalize and map from (N, 3, d1, d2) to (N, d1, d2, 3)
        np_images = np.transpose(np_images, (0, 2, 3, 1))
        np_images = np_images * std + mean

    plt.imshow(np_images.reshape((np_images.shape[0] * np_images.shape[1], np_images.shape[2], np_images.shape[3])))
    plt.show()


def visualize(loader, colors, std, mean):
    # Visualize a whole batch
    temp = iter(loader)
    images, labels = temp.next()
    show(images=images, is_label=False, colors=colors, std=std, mean=mean)
    show(images=labels, is_label=True, colors=colors, std=std, mean=mean)


# Save model checkpoints (supports amp)
def save_checkpoint(net, optimizer, lr_scheduler, filename='temp.pt'):
    checkpoint = {
        'model': net.state_dict(),
        'optimizer': optimizer.state_dict() if optimizer is not None else None,
        'lr_scheduler': lr_scheduler.state_dict() if lr_scheduler is not None else None
    }
    torch.save(checkpoint, filename)


# Load model checkpoints (supports amp)
def load_checkpoint(net, optimizer, lr_scheduler, filename):
    checkpoint = torch.load(filename)
    net.load_state_dict(checkpoint['model'])

    # print(len(net.state_dict().keys()))
    # print(len(checkpoint.keys()))
    # ori_state_dict = net.state_dict()
    # for key in checkpoint.keys():
    #     ori_state_dict[key[7:]] = checkpoint[key]
    # net.load_state_dict(ori_state_dict)

    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer'])
    if lr_scheduler is not None:
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])


def init(batch_size, state, input_sizes, std, mean, dataset, city_aug=0):
    # Return data_loaders
    # depending on whether the state is
    # 1: training
    # 2: just testing

    # Transformations
    # ! Can't use torchvision.Transforms.Compose
    if dataset == 'voc':
        base = base_voc
        workers = 4
        transform_train = Compose(
            [ToTensor(),
             # RandomResize(min_size=input_sizes[0], max_size=input_sizes[1]),
             RandomScale(min_scale=0.5, max_scale=1.5),
             RandomCrop(size=input_sizes[0]),
             RandomHorizontalFlip(flip_prob=0.5),
             Normalize(mean=mean, std=std)])
        transform_test = Compose(
            [ToTensor(),
             ZeroPad(size=input_sizes[2]),
             Normalize(mean=mean, std=std)])
    elif dataset == 'city' or dataset == 'gtav' or dataset == 'synthia':  # All the same size
        if dataset == 'city':
            base = base_city
        elif dataset == 'gtav':
            base = base_gtav
        else:
            base = base_synthia
        outlier = False if dataset == 'city' else True  # GTAV has fucked up label ID
        workers = 8

        if city_aug == 3:  # SYNTHIA & GTAV
            if dataset == 'gtav':
                transform_train = Compose(
                    [ToTensor(),
                     Resize(size_label=input_sizes[1], size_image=input_sizes[1]),
                     RandomCrop(size=input_sizes[0]),
                     RandomHorizontalFlip(flip_prob=0.5),
                     Normalize(mean=mean, std=std),
                     LabelMap(label_id_map_city, outlier=outlier)])
            else:
                transform_train = Compose(
                    [ToTensor(),
                     RandomCrop(size=input_sizes[0]),
                     RandomHorizontalFlip(flip_prob=0.5),
                     Normalize(mean=mean, std=std),
                     LabelMap(label_id_map_synthia, outlier=outlier)])
            transform_test = Compose(
                [ToTensor(),
                 Resize(size_image=input_sizes[2], size_label=input_sizes[2]),
                 Normalize(mean=mean, std=std),
                 LabelMap(label_id_map_city)])
        elif city_aug == 2:  # ERFNet
            transform_train = Compose(
                [ToTensor(),
                 Resize(size_image=input_sizes[0], size_label=input_sizes[0]),
                 LabelMap(label_id_map_city, outlier=outlier),
                 RandomTranslation(trans_h=2, trans_w=2),
                 RandomHorizontalFlip(flip_prob=0.5)])
            transform_test = Compose(
                [ToTensor(),
                 Resize(size_image=input_sizes[0], size_label=input_sizes[2]),
                 LabelMap(label_id_map_city)])
        elif city_aug == 1:  # City big
            transform_train = Compose(
                [ToTensor(),
                 RandomCrop(size=input_sizes[0]),
                 LabelMap(label_id_map_city, outlier=outlier),
                 RandomTranslation(trans_h=2, trans_w=2),
                 RandomHorizontalFlip(flip_prob=0.5),
                 Normalize(mean=mean, std=std)])
            transform_test = Compose(
                [ToTensor(),
                 Resize(size_image=input_sizes[2], size_label=input_sizes[2]),
                 Normalize(mean=mean, std=std),
                 LabelMap(label_id_map_city)])
        else:  # Standard city
            transform_train = Compose(
                [ToTensor(),
                 # RandomResize(min_size=input_sizes[0], max_size=input_sizes[1]),
                 RandomScale(min_scale=0.5, max_scale=1.5),
                 RandomCrop(size=input_sizes[0]),
                 RandomHorizontalFlip(flip_prob=0.5),
                 Normalize(mean=mean, std=std),
                 LabelMap(label_id_map_city, outlier=outlier)])
            transform_test = Compose(
                [ToTensor(),
                 Resize(size_image=input_sizes[2], size_label=input_sizes[2]),
                 Normalize(mean=mean, std=std),
                 LabelMap(label_id_map_city)])
    else:
        raise ValueError

    # Not the actual test set (i.e. validation set)
    test_set = StandardSegmentationDataset(root=base_city if dataset == 'gtav' or dataset == 'synthia' else base,
                                           image_set='val', transforms=transform_test,
                                           data_set='city' if dataset == 'gtav' or dataset == 'synthia' else dataset)
    if city_aug == 1 or city_aug == 3:
        val_loader = torch.utils.data.DataLoader(dataset=test_set, batch_size=2, num_workers=workers, shuffle=False)
    else:
        val_loader = torch.utils.data.DataLoader(dataset=test_set, batch_size=batch_size, num_workers=workers,
                                                 shuffle=False)

    # Testing
    if state == 1:
        return val_loader
    else:
        # Training
        train_set = StandardSegmentationDataset(root=base, image_set='trainaug' if dataset == 'voc' else 'train',
                                                transforms=transform_train, data_set=dataset)
        train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=batch_size,
                                                   num_workers=workers, shuffle=True)
        return train_loader, val_loader


def train_schedule(writer, loader, val_num_steps, validation_loader, device, criterion, net, optimizer, lr_scheduler,
                   num_epochs, is_mixed_precision, num_classes, categories, input_sizes, classes):
    # Poly training schedule
    # Validate and find the best snapshot
    best_mIoU = 0
    net.train()
    epoch = 0
    loss_num_steps = int(len(loader) / 10)
    if is_mixed_precision:
        scaler = GradScaler()

    # Training
    while epoch < num_epochs:
        net.train()
        conf_mat = ConfusionMatrix(num_classes)
        running_loss = 0.0
        time_now = time.time()
        for i, data in enumerate(loader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()

            with autocast(is_mixed_precision):
                outputs = net(inputs)['out']
                outputs = torch.nn.functional.interpolate(outputs, size=input_sizes[0], mode='bilinear', align_corners=True)
                conf_mat.update(labels.flatten(), outputs.argmax(1).flatten())
                loss = criterion(outputs, labels)

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

            # Validate and find the best snapshot
            if current_step_num % val_num_steps == (val_num_steps - 1):
                test_pixel_accuracy, test_mIoU = test_one_set(loader=validation_loader, device=device, net=net,
                                                              num_classes=num_classes, categories=categories,
                                                              output_size=input_sizes[2], classes=classes,
                                                              is_mixed_precision=is_mixed_precision)
                writer.add_scalar('test pixel accuracy',
                                  test_pixel_accuracy,
                                  current_step_num)
                writer.add_scalar('test mIoU',
                                  test_mIoU,
                                  current_step_num)
                net.train()

                # Record best model (straight to disk)
                if test_mIoU > best_mIoU:
                    best_mIoU = test_mIoU
                    save_checkpoint(net=net, optimizer=optimizer, lr_scheduler=lr_scheduler)

        # Evaluate training accuracies (same metric as validation, but must be on-the-fly to save time)
        with autocast(is_mixed_precision):
            acc_global, acc, iu = conf_mat.compute()
        print(categories)
        print((
            'global correct: {:.2f}\n'
            'average row correct: {}\n'
            'IoU: {}\n'
            'mean IoU: {:.2f}').format(
            acc_global.item() * 100,
            ['{:.2f}'.format(i) for i in (acc * 100).tolist()],
            ['{:.2f}'.format(i) for i in (iu * 100).tolist()],
            iu.mean().item() * 100))

        train_pixel_acc = acc_global.item() * 100
        train_mIoU = iu.mean().item() * 100
        writer.add_scalar('train pixel accuracy',
                          train_pixel_acc,
                          epoch + 1)
        writer.add_scalar('train mIoU',
                          train_mIoU,
                          epoch + 1)

        epoch += 1
        print('Epoch time: %.2fs' % (time.time() - time_now))


# Copied and modified from torch/vision/references/segmentation
def test_one_set(loader, device, net, num_classes, categories, output_size, is_mixed_precision, classes=None):
    # Evaluate on 1 data_loader
    net.eval()
    conf_mat = ConfusionMatrix(num_classes)
    with torch.no_grad():
        for image, target in tqdm(loader):
            image, target = image.to(device), target.to(device)
            with autocast(is_mixed_precision):
                output = net(image)['out']
                output = torch.nn.functional.interpolate(output, size=output_size, mode='bilinear', align_corners=True)
                conf_mat.update(target.flatten(), output.argmax(1).flatten())

    acc_global, acc, iu = conf_mat.compute()
    print(categories)
    print((
            'global correct: {:.2f}\n'
            'average row correct: {}\n'
            'IoU: {}\n'
            'mean IoU: {:.2f}\n'
            'mean IoU-16: {:.2f}\n'
            'mean IoU-13: {:.2f}').format(
                acc_global.item() * 100,
                ['{:.2f}'.format(i) for i in (acc * 100).tolist()],
                ['{:.2f}'.format(i) for i in (iu * 100).tolist()],
                iu.mean().item() * 100,
                iu[iou_16].mean().item() * 100,
                iu[iou_13].mean().item() * 100))

    iou = iu.mean().item() * 100
    if classes == 16:
        iou = iu[iou_16].mean().item() * 100
    elif classes == 13:
        iou = iu[iou_13].mean().item() * 100

    return acc_global.item() * 100, iou
