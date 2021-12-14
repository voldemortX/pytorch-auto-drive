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
    categories = [
        'Background',
        'Aeroplane', 'Bicycle', 'Bird', 'Boat',
        'Bottle', 'Bus', 'Car', 'Cat',
        'Chair', 'Cow', 'Diningtable', 'Dog',
        'Horse', 'Motorbike', 'Person', 'Pottedplant',
        'Sheep', 'Sofa', 'Train', 'Tvmonitor'
    ]

    colors = [
        [0, 0, 0],
        [128, 0, 0], [0, 128, 0], [128, 128, 0], [0, 0, 128],
        [128, 0, 128], [0, 128, 128], [128, 128, 128], [64, 0, 0],
        [192, 0, 0], [64, 128, 0], [192, 128, 0], [64, 0, 128],
        [192, 0, 128], [64, 128, 128], [192, 128, 128], [0, 64, 0],
        [128, 64, 0], [0, 192, 0], [128, 192, 0], [0, 64, 128],
        [255, 255, 255]  # last color for ignore
    ]

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
    categories = [
        'road', 'sidewalk', 'building', 'wall',
        'fence', 'pole', 'traffic light', 'traffic sign',
        'vegetation', 'terrain', 'sky', 'person',
        'rider', 'car', 'truck', 'bus',
        'train', 'motorcycle', 'bicycle'
    ]

    colors = [
        [128, 64, 128], [244, 35, 232], [70, 70, 70], [102, 102, 156],
        [190, 153, 153], [153, 153, 153], [250, 170, 30], [220, 220, 0],
        [107, 142, 35], [152, 251, 152], [70, 130, 180], [220, 20, 60],
        [255, 0, 0], [0, 0, 142], [0, 0, 70], [0, 60, 100],
        [0, 80, 100], [0, 0, 230], [119, 11, 32],
        [0, 0, 0]  # last color for ignore
    ]

    cities = [
        'aachen', 'bremen', 'darmstadt', 'erfurt', 'hanover',
        'krefeld', 'strasbourg', 'tubingen', 'weimar', 'bochum',
        'cologne', 'dusseldorf', 'hamburg', 'jena', 'monchengladbach',
        'stuttgart', 'ulm', 'zurich'
    ]

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
class GTAV_Segmentation(CityscapesSegmentation):
    cities = None

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
class SYNTHIA_Segmentation(CityscapesSegmentation):
    cities = None

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
