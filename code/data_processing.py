import torchvision
import torch
import os
import numpy as np
from PIL import Image

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
colors = [[0, 0, 0],
          [128, 0, 0], [0, 128, 0], [128, 128, 0], [0, 0, 128],
          [128, 0, 128], [0, 128, 128], [128, 128, 128], [64, 0, 0],
          [192, 0, 0], [64, 128, 0], [192, 128, 0], [64, 0, 128],
          [192, 0, 128], [64, 128, 128], [192, 128, 128], [0, 64, 0],
          [128, 64, 0], [0, 192, 0], [128, 192, 0], [0, 64, 128],
          [255, 255, 255]]


class MaskToTensor(object):
    #  For {0~20, 255} style masks(! do not use ToTensor) => H x W tensor
    def __call__(self, mask):
        target = np.array(mask, copy=False)
        target = torch.from_numpy(target.reshape(target.shape[0], target.shape[1])).long()
        return target


# Reimplemented and simplified based on torchvision.datasets.VOCSegmentation
class StandardSegmentationDataset(torchvision.datasets.VisionDataset):
    def __init__(self, root, image_set, transforms=None, transform=None, target_transform=None):
        super().__init__(root, transforms, transform, target_transform)
        image_dir = os.path.join(root, 'JPEGImages')
        mask_dir = os.path.join(root, 'SegmentationClassAug')
        splits_dir = os.path.join(root, 'ImageSets/Segmentation')
        split_f = os.path.join(splits_dir, image_set + '.txt')

        with open(os.path.join(split_f), "r") as f:
            file_names = [x.strip() for x in f.readlines()]

        self.images = [os.path.join(image_dir, x + ".jpg") for x in file_names]
        self.masks = [os.path.join(mask_dir, x + ".png") for x in file_names]
        assert (len(self.images) == len(self.masks))

    def __getitem__(self, index):
        # Return x(input image) & y(mask image)
        img = Image.open(self.images[index]).convert('RGB')
        target = Image.open(self.masks[index])

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.images)
