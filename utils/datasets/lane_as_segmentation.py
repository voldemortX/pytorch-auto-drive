import os
import torch
from PIL import Image
from torchvision.datasets import VisionDataset

from .builder import DATASETS


# Lane detection as segmentation
class _StandardLaneDetectionDataset(VisionDataset):
    def __init__(self, root, image_set, transforms=None):
        super().__init__(root, transforms, None, None)
        if image_set == 'valfast':
            self.test = 1
        elif image_set == 'test' or image_set == 'val':  # Different format (without lane existence annotations)
            self.test = 2
        else:
            self.test = 0

        self.init_dataset(root)
        self.image_set = image_set
        self.splits_dir = os.path.join(root, 'lists')

        self._init_all()

        assert (len(self.images) == len(self.masks))

    def init_dataset(self, root):
        raise NotImplementedError

    def __getitem__(self, index):
        # Return x (input image) & y (mask image, i.e. pixel-wise supervision) & lane existence (a list),
        # if not just testing,
        # else just return input image.
        img = Image.open(self.images[index]).convert('RGB')
        if self.test == 2:
            target = self.masks[index]
        elif self.test == 1:
            target = Image.open(self.masks[index])
        else:
            target = Image.open(self.masks[index])
            lane_existence = torch.tensor(self.lane_existences[index]).float()

        # Transforms
        if self.transforms is not None:
            img, target = self.transforms(img, target)
        if self.test > 0:
            return img, target
        else:
            return img, target, lane_existence

    def __len__(self):
        return len(self.images)

    def _init_all(self):
        # Got the lists from 2 datasets to be in the same format
        split_f = os.path.join(self.splits_dir, self.image_set + '.txt')
        with open(split_f, "r") as f:
            contents = [x.strip() for x in f.readlines()]
        if self.test == 2:  # Test
            self.images = [os.path.join(self.image_dir, x + self.image_suffix) for x in contents]
            self.masks = [os.path.join(self.output_prefix, x + self.output_suffix) for x in contents]
        elif self.test == 1:  # Val
            self.images = [os.path.join(self.image_dir, x[:x.find(' ')] + self.image_suffix) for x in contents]
            self.masks = [os.path.join(self.mask_dir, x[:x.find(' ')] + '.png') for x in contents]
        else:  # Train
            self.images = [os.path.join(self.image_dir, x[:x.find(' ')] + self.image_suffix) for x in contents]
            self.masks = [os.path.join(self.mask_dir, x[:x.find(' ')] + '.png') for x in contents]
            self.lane_existences = [list(map(int, x[x.find(' '):].split())) for x in contents]


# TuSimple
@DATASETS.register()
class TuSimpleAsSegmentation(_StandardLaneDetectionDataset):
    colors = [
        [0, 0, 0],  # background
        [255, 0, 255], [0, 255, 0], [0, 0, 255], [255, 0, 0], [255, 255, 0], [0, 255, 255],
        [0, 0, 0]  # ignore
    ]

    def init_dataset(self, root):
        self.image_dir = os.path.join(root, 'clips')
        self.mask_dir = os.path.join(root, 'segGT6')
        self.output_prefix = 'clips'
        self.output_suffix = '.jpg'
        self.image_suffix = '.jpg'


# CULane
@DATASETS.register()
class CULaneAsSegmentation(_StandardLaneDetectionDataset):
    colors = [
        [0, 0, 0],  # background
        [0, 255, 0], [0, 0, 255], [255, 0, 0], [255, 255, 0],
        [0, 0, 0]  # ignore
    ]

    def init_dataset(self, root):
        self.image_dir = root
        self.mask_dir = os.path.join(root, 'laneseg_label_w16')
        self.output_prefix = './output'
        self.output_suffix = '.lines.txt'
        self.image_suffix = '.jpg'
        if not os.path.exists(self.output_prefix):
            os.makedirs(self.output_prefix)


# LLAMAS
@DATASETS.register()
class LLAMAS_AsSegmentation(CULaneAsSegmentation):
    def init_dataset(self, root):
        self.image_dir = os.path.join(root, 'color_images')
        self.mask_dir = os.path.join(root, 'laneseg_labels')
        self.output_prefix = './output'
        self.output_suffix = '.lines.txt'
        self.image_suffix = '.png'
        if not os.path.exists(self.output_prefix):
            os.makedirs(self.output_prefix)
