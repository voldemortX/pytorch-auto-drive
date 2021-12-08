import os
import numpy as np
from PIL import Image
from torchvision.datasets import VisionDataset

from .builder import DATASETS


# Reimplemented based on torchvision.datasets.VOCSegmentation
class _StandardSegmentationDataset(VisionDataset):
    def __init__(self, root, image_set, transforms=None, mask_type='.png'):
        super().__init__(root, transforms, None, None)
        self.mask_type = mask_type
        self.images = self.masks = []
        self.init_dataset(root, image_set)
        assert (len(self.images) == len(self.masks))

    def init_dataset(self, root, image_set):
        raise NotImplementedError

    def __getitem__(self, index):
        img = Image.open(self.images[index]).convert('RGB')
        # Return x(input image) & y(mask images as a list)
        # Supports .png & .npy
        target = Image.open(self.masks[index]) if '.png' in self.masks[index] else np.load(self.masks[index])

        # Transforms
        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.images)


# VOC
@DATASETS.register()
class PASCAL_VOC_Segmentation(_StandardSegmentationDataset):
    def init_dataset(self, root, image_set):
        image_dir = os.path.join(root, 'JPEGImages')
        mask_dir = os.path.join(root, 'SegmentationClassAug')
        splits_dir = os.path.join(root, 'ImageSets/Segmentation')
        split_f = os.path.join(splits_dir, image_set + '.txt')
        with open(split_f, "r") as f:
            file_names = [x.strip() for x in f.readlines()]

        self.images = [os.path.join(image_dir, x + ".jpg") for x in file_names]
        self.masks = [os.path.join(mask_dir, x + self.mask_type) for x in file_names]


# Cityscapes
@DATASETS.register()
class CityscapesSegmentation(_StandardSegmentationDataset):
    def init_dataset(self, root, image_set):
        image_dir = os.path.join(root, 'leftImg8bit')
        mask_dir = os.path.join(root, 'gtFine')

        if image_set == 'val':
            image_dir = os.path.join(image_dir, image_set)
            mask_dir = os.path.join(mask_dir, image_set)
        else:
            image_dir = os.path.join(image_dir, 'train')
            mask_dir = os.path.join(mask_dir, 'train')

        # We first generate data lists before all this, so we can do this easier
        splits_dir = os.path.join(root, 'data_lists')
        split_f = os.path.join(splits_dir, image_set + '.txt')
        with open(split_f, "r") as f:
            file_names = [x.strip() for x in f.readlines()]

        self.images = [os.path.join(image_dir, x + "_leftImg8bit.png") for x in file_names]
        self.masks = [os.path.join(mask_dir, x + "_gtFine_labelIds" + self.mask_type) for x in file_names]


# GTAV
@DATASETS.register()
class GTAV_Segmentation(_StandardSegmentationDataset):
    def init_dataset(self, root, image_set):
        image_dir = os.path.join(root, 'images')
        mask_dir = os.path.join(root, 'labels')

        # We first generate data lists before all this, so we can do this easier
        splits_dir = os.path.join(root, 'data_lists')
        split_f = os.path.join(splits_dir, image_set + '.txt')
        with open(split_f, "r") as f:
            file_names = [x.strip() for x in f.readlines()]

        self.images = [os.path.join(image_dir, x + ".png") for x in file_names]
        self.masks = [os.path.join(mask_dir, x + self.mask_type) for x in file_names]


# SYNTHIA
@DATASETS.register()
class SYNTHIA_Segmentation(_StandardSegmentationDataset):
    def init_dataset(self, root, image_set):
        image_dir = os.path.join(root, 'RGB', image_set)
        mask_dir = os.path.join(root, 'GT/LABELS_CONVERTED', image_set)

        # We first generate data lists before all this, so we can do this easier
        splits_dir = os.path.join(root, 'data_lists')
        split_f = os.path.join(splits_dir, image_set + '.txt')
        with open(split_f, "r") as f:
            file_names = [x.strip() for x in f.readlines()]

        self.images = [os.path.join(image_dir, x + ".png") for x in file_names]
        self.masks = [os.path.join(mask_dir, x + self.mask_type) for x in file_names]
