import torch
import torchvision
import time
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from PIL import Image
from apex import amp
from data_processing import StandardSegmentationDataset, MaskToTensor, colors, mean, std
# Input_size = (513, 513)
Input_size = (128, 128)  # In order to test it on GTX-960M(2G memory)
N_classes = 21


def deeplab_v3():
    # Define deeplabV3 with ResNet101(With only ImageNet pretraining)
    return torchvision.models.segmentation.deeplabv3_resnet101(pretrained=False, num_classes=N_classes)


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


def show(images, is_label):
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


def visualize(loader, categories):
    # Visualize a whole batch
    temp = iter(loader)
    images, labels = temp.next()
    show(images=images, is_label=False)
    show(images=labels, is_label=True)


# Save model checkpoints(supports amp)
def save_checkpoint(net, optimizer, lr_scheduler, is_mixed_precision, filename='temp.pt'):
    checkpoint = {
        'model': net.state_dict(),
        'optimizer': optimizer.state_dict(),
        'lr_scheduler': lr_scheduler.state_dict(),
        'amp': amp.state_dict() if is_mixed_precision else None
    }
    torch.save(checkpoint, filename)


# Load model checkpoints(supports amp)
def load_checkpoint(net, optimizer, lr_scheduler, is_mixed_precision, filename):
    checkpoint = torch.load(filename)
    net.load_state_dict(checkpoint['net'])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer'])
    if lr_scheduler is not None:
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
    if is_mixed_precision:
        amp.load_state_dict(checkpoint['amp'])


def init(batch_size, state):
    # Return data_loaders/data_loader
    # depending on whether this is
    # 0: initial training(trainaug, val)
    # 1: last training(trainval)
    # 2: final test("test", which is not available)
    #base = '../data/VOCtrainval_11-May-2012/VOCdevkit/VOC2012'
    base = '../data_test/VOCtrainval_11-May-2012/VOCdevkit/VOC2012'
    categories = []

    # Transformations
    # Statistics for ImageNet pretrained models
    normalization_pretrained = torchvision.transforms.Normalize(mean=mean, std=std)
    transform_train_target = torchvision.transforms.Compose(
        [torchvision.transforms.Resize(size=Input_size, interpolation=Image.NEAREST),
         MaskToTensor()
         ])
    transform_test_target = torchvision.transforms.Compose(
        [torchvision.transforms.Resize(size=Input_size, interpolation=Image.NEAREST),
         MaskToTensor()
         ])
    transform_train = torchvision.transforms.Compose(
        [torchvision.transforms.Resize(size=Input_size, interpolation=Image.BILINEAR),
         torchvision.transforms.ToTensor(),
         normalization_pretrained])
    transform_test = torchvision.transforms.Compose(
        [torchvision.transforms.Resize(size=Input_size, interpolation=Image.BILINEAR),
         torchvision.transforms.ToTensor(),
         normalization_pretrained])

    # Not the actual test set(i.e. validation set)
    if state == 2:
        test_set = StandardSegmentationDataset(root=base, image_set='val', transform=transform_test,
                                               target_transform=transform_test_target)
        test_loader = torch.utils.data.DataLoader(dataset=test_set, batch_size=batch_size,
                                                  num_workers=4, shuffle=False)
        return test_loader, categories

    # trainval
    elif state == 1:
        train_set = StandardSegmentationDataset(root=base, image_set='trainval', transform=transform_train,
                                                target_transform=transform_train_target)
        train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=batch_size,
                                                   num_workers=4, shuffle=True)
        return train_loader, categories

    # train, val
    elif state == 0:
        train_set = StandardSegmentationDataset(root=base, image_set='trainaug', transform=transform_train,
                                                target_transform=transform_train_target)
        train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=batch_size,
                                                   num_workers=4, shuffle=True)
        val_set = StandardSegmentationDataset(root=base, image_set='val', transform=transform_test,
                                              target_transform=transform_test_target)
        val_loader = torch.utils.data.DataLoader(dataset=val_set, batch_size=batch_size,
                                                 num_workers=4, shuffle=False)
        return train_loader, val_loader, categories


def train_schedule(writer, loader, with_validation, validation_loader, device, criterion, net, optimizer, lr_scheduler,
                   num_epochs, is_mixed_precision):
    # Poly training schedule
    # Validate and find the best snapshot
    if with_validation:
        best_mIoU = 0

    net.train()
    epoch = 0

    # Training
    while epoch < num_epochs:
        net.train()
        running_loss = 0.0
        time_now = time.time()
        for i, data in enumerate(loader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = net(inputs)['out']
            loss = criterion(outputs, labels)
            if is_mixed_precision:
                # 2/3 & 3/3 of mixed precision training with amp
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()
            optimizer.step()
            lr_scheduler.step()
            running_loss += loss.item()
            if i % 100 == 99:
                print('[%d, %d] loss: %.4f' % (epoch + 1, i + 1, running_loss / 100))
                writer.add_scalar('training loss',
                                  running_loss / 100,
                                  epoch * len(loader) + i + 1)
                running_loss = 0.0

        # Evaluate training accuracies(same metric as validation)
        train_pixel_acc, train_mIoU = test_one_set(loader=loader, device=device, net=net)
        writer.add_scalar('train pixel accuracy',
                          train_pixel_acc,
                          epoch + 1)
        writer.add_scalar('train mIoU',
                          train_mIoU,
                          epoch + 1)
        print('Epoch time: %.2fs' % (time.time() - time_now))
    
        # Validate and find the best snapshot
        if with_validation:
            test_pixel_accuracy, test_mIoU = test_one_set(loader=validation_loader, device=device, net=net)
            writer.add_scalar('test pixel accuracy',
                              test_pixel_accuracy,
                              epoch + 1)
            writer.add_scalar('test mIoU',
                              test_mIoU,
                              epoch + 1)

            # Record best model(Straight to disk)
            if test_mIoU > best_mIoU:
                best_mIoU = test_mIoU
                save_checkpoint(net=net, optimizer=optimizer, lr_scheduler=lr_scheduler,
                                is_mixed_precision=is_mixed_precision)

        epoch += 1

    # Save just the last states
    if not with_validation:
        save_checkpoint(net=net, optimizer=optimizer, lr_scheduler=lr_scheduler,
                        is_mixed_precision=is_mixed_precision)


# Copied and modified from torch/vision/references/segmentation
def test_one_set(loader, device, net):
    # Evaluate on 1 data_loader
    net.eval()
    conf_mat = ConfusionMatrix(N_classes)
    with torch.no_grad():
        for image, target in tqdm(loader):
            image, target = image.to(device), target.to(device)
            output = net(image)['out']
            conf_mat.update(target.flatten(), output.argmax(1).flatten())

    acc_global, acc, iu = conf_mat.compute()
    print((
            'global correct: {:.1f}\n'
            'average row correct: {}\n'
            'IoU: {}\n'
            'mean IoU: {:.1f}').format(
                acc_global.item() * 100,
                ['{:.1f}'.format(i) for i in (acc * 100).tolist()],
                ['{:.1f}'.format(i) for i in (iu * 100).tolist()],
                iu.mean().item() * 100))

    return acc_global.item() * 100, iu.mean().item() * 100
