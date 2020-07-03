import torch
import time
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from apex import amp
from torchvision_models.segmentation import erfnet_resnet
from data_processing import StandardLaneDetectionDataset
from transforms import ToTensor, Normalize, RandomHorizontalFlip, Resize, RandomResize, RandomCrop, RandomTranslation,\
                       ZeroPad, LabelMap, Compose


def erfnet_tusimple(num_classes, pretrained_weights='erfnet_encoder_pretrained.pth.tar'):
    # Define ERFNet for TuSimple (With only ImageNet pretraining)
    return erfnet_resnet(pretrained_weights=pretrained_weights, num_classes=num_classes, aux=6,
                         dropout_1=0.1, dropout_2=0.1, flattened_size=4500)


def erfnet_culane(num_classes, pretrained_weights='erfnet_encoder_pretrained.pth.tar'):
    # Define ERFNet for CULane (With only ImageNet pretraining)
    return erfnet_resnet(pretrained_weights=pretrained_weights, num_classes=num_classes, aux=4,
                         dropout_1=0.1, dropout_2=0.1, flattened_size=3965)


def init(batch_size, state, input_sizes, dataset):
    # Return data_loaders
    # depending on whether the state is
    # 1: training
    # 2: just testing
    # 3: submit to server

    pass


def train_schedule(writer, loader, val_num_steps, validation_loader, device, criterion, net, optimizer, lr_scheduler,
                   num_epochs, is_mixed_precision, num_classes, categories, input_sizes):
    pass


def test_one_set():
    # Evaluate on 1 data_loader
    pass
