import torchvision
import os
import torch
import numpy as np
from PIL import Image


mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

# Here 'training resize min' is also the final training crop size as RandomResize & RandomCrop are used together
# For PASCAL VOC 2012
base_voc = '../data/VOCtrainval_11-May-2012/VOCdevkit/VOC2012'
sizes_voc = [(321, 321), (505, 505), (505, 505)]  # training resize min/training resize max/testing label size
num_classes_voc = 21
colors_voc = [[0, 0, 0],
              [128, 0, 0], [0, 128, 0], [128, 128, 0], [0, 0, 128],
              [128, 0, 128], [0, 128, 128], [128, 128, 128], [64, 0, 0],
              [192, 0, 0], [64, 128, 0], [192, 128, 0], [64, 0, 128],
              [192, 0, 128], [64, 128, 128], [192, 128, 128], [0, 64, 0],
              [128, 64, 0], [0, 192, 0], [128, 192, 0], [0, 64, 128],
              [255, 255, 255]]
categories_voc = ['Background',
                  'Aeroplane', 'Bicycle', 'Bird', 'Boat',
                  'Bottle', 'Bus', 'Car', 'Cat',
                  'Chair', 'Cow', 'Diningtable', 'Dog',
                  'Horse', 'Motorbike', 'Person', 'Pottedplant',
                  'Sheep', 'Sofa', 'Train', 'Tvmonitor']

# For cityscapes (19 classes, ignore as black, no such thing as background)
base_city = '../../../dataset/cityscapes'
sizes_city = [(257, 513), (513, 1025), (513, 1025)]  # training resize min/training resize max/testing label size
sizes_city_big = [(512, 1024), (512, 1024), (1024, 2048)]  # training resize min/training resize max/testing label size
sizes_city_erfnet = [(512, 1024), (64, 128), (512, 1024)]  # input/encoder output/testing label size
weights_city_erfnet = [2.8149201869965, 6.9850029945374, 3.7890393733978, 9.9428062438965,
                       9.7702074050903, 9.5110931396484, 10.311357498169, 10.026463508606,
                       4.6323022842407, 9.5608062744141, 7.8698215484619, 9.5168733596802,
                       10.373730659485, 6.6616044044495, 10.260489463806, 10.287888526917,
                       10.289801597595, 10.405355453491, 10.138095855713]
num_classes_city = 19
colors_city = [
               [128, 64, 128], [244, 35, 232], [70, 70, 70], [102, 102, 156],
               [190, 153, 153], [153, 153, 153], [250, 170, 30], [220, 220, 0],
               [107, 142, 35], [152, 251, 152], [70, 130, 180], [220, 20, 60],
               [255, 0, 0], [0, 0, 142], [0, 0, 70], [0, 60, 100],
               [0, 80, 100], [0, 0, 230], [119, 11, 32],
               [0, 0, 0]]
categories_city = [
                   'road', 'sidewalk', 'building', 'wall',
                   'fence', 'pole', 'traffic light', 'traffic sign',
                   'vegetation', 'terrain', 'sky', 'person',
                   'rider', 'car', 'truck', 'bus',
                   'train', 'motorcycle', 'bicycle']
label_id_map_city = [255, 255, 255, 255, 255, 255, 255,
                     0,   1,   255, 255, 2,   3,   4,
                     255, 255, 255, 5,   255, 6,   7,
                     8,   9,   10,  11,  12,  13,  14,
                     15,  255, 255, 16,  17,  18]
train_cities = ['aachen', 'bremen', 'darmstadt', 'erfurt', 'hanover',
                'krefeld', 'strasbourg', 'tubingen', 'weimar', 'bochum',
                'cologne', 'dusseldorf', 'hamburg', 'jena', 'monchengladbach',
                'stuttgart', 'ulm', 'zurich']

# For GTAV (19 classes, ignore as black, no such thing as background)
base_gtav = '../../../dataset/gtav'
sizes_gtav = [(257, 513), (513, 1025), (513, 1025)]  # training resize min/training resize max/testing label size
sizes_gtav_erfnet = [(512, 1024), (64, 128), (512, 1024)]  # input/encoder output/testing label size

# For TuSimple
base_tusimple = '../../../dataset/tusimple'
sizes_tusimple = [(256, 512), (720, 1280)]  # training size/actual size
num_classes_tusimple = 7
weights_tusimple = [0.4, 1, 1, 1, 1, 1, 1]


# For CULane
base_culane = '../../../dataset/culane'
sizes_culane = [(288, 800), (590, 1640)]  # training size/actual size
num_classes_culane = 5
weights_culane = [0.4, 1, 1, 1, 1]


# Reimplemented based on torchvision.datasets.VOCSegmentation
class StandardSegmentationDataset(torchvision.datasets.VisionDataset):
    def __init__(self, root, image_set, transforms=None, transform=None, target_transform=None, data_set='voc',
                 mask_type='.png'):
        super().__init__(root, transforms, transform, target_transform)
        self.mask_type = mask_type
        if data_set == 'voc':
            self._voc_init(root, image_set)
        elif data_set == 'city':
            self._city_init(root, image_set)
        elif data_set == 'gtav':
            self._gtav_init(root, image_set)
        else:
            raise ValueError

        assert (len(self.images) == len(self.masks))

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

    def _voc_init(self, root, image_set):
        image_dir = os.path.join(root, 'JPEGImages')
        mask_dir = os.path.join(root, 'SegmentationClassAug')
        splits_dir = os.path.join(root, 'ImageSets/Segmentation')
        split_f = os.path.join(splits_dir, image_set + '.txt')
        with open(os.path.join(split_f), "r") as f:
            file_names = [x.strip() for x in f.readlines()]

        self.images = [os.path.join(image_dir, x + ".jpg") for x in file_names]
        self.masks = [os.path.join(mask_dir, x + self.mask_type) for x in file_names]

    def _city_init(self, root, image_set):
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
        with open(os.path.join(split_f), "r") as f:
            file_names = [x.strip() for x in f.readlines()]

        self.images = [os.path.join(image_dir, x + "_leftImg8bit.png") for x in file_names]
        self.masks = [os.path.join(mask_dir, x + "_gtFine_labelIds" + self.mask_type) for x in file_names]

    def _gtav_init(self, root, image_set):
        image_dir = os.path.join(root, 'images')
        mask_dir = os.path.join(root, 'labels')

        # We first generate data lists before all this, so we can do this easier
        splits_dir = os.path.join(root, 'data_lists')
        split_f = os.path.join(splits_dir, image_set + '.txt')
        with open(os.path.join(split_f), "r") as f:
            file_names = [x.strip() for x in f.readlines()]

        self.images = [os.path.join(image_dir, x + ".png") for x in file_names]
        self.masks = [os.path.join(mask_dir, x + self.mask_type) for x in file_names]


class StandardLaneDetectionDataset(torchvision.datasets.VisionDataset):
    def __init__(self, root, image_set, transforms=None, transform=None, target_transform=None, data_set='tusimple'):
        super().__init__(root, transforms, transform, target_transform)
        self.is_test = (image_set == 'test' or image_set == 'val')
        image_dir, splits_dir, mask_dir = self.get_init_parameters(root=root, data_set=data_set)
        self._init_all(image_dir=image_dir, splits_dir=splits_dir, mask_dir=mask_dir, image_set=image_set)

        assert (len(self.images) == len(self.masks))

    def __getitem__(self, index):
        # Return x (input image) & y (mask image, i.e. pixel-wise supervision) & lane existence (a list),
        # if not just testing,
        # else just return input image.
        img = Image.open(self.images[index]).convert('RGB')
        if self.is_test:
            target = ''  # To accommodate transforms
        else:
            target = Image.open(self.masks[index])
            lane_existence = torch.tensor(self.lane_existences[index]).float()

        # Transforms
        if self.transforms is not None:
            img, target = self.transforms(img, target)

        if self.is_test:
            return img
        else:
            return img, target, lane_existence

    def __len__(self):
        return len(self.images)

    @staticmethod
    def get_init_parameters(root, data_set):
        if data_set == 'tusimple':
            image_dir = os.path.join(root, 'clips')
            mask_dir = os.path.join(root, 'segGT6')
        elif data_set == 'culane':
            image_dir = root
            mask_dir = os.path.join(root, 'laneseg_label_w16')
        else:
            raise ValueError

        splits_dir = os.path.join(root, 'lists')

        return image_dir, splits_dir, mask_dir

    def _init_all(self, image_dir, splits_dir, mask_dir, image_set):
        # Got the lists from 2 datasets to be in the same format
        split_f = os.path.join(splits_dir, image_set + '.txt')
        with open(os.path.join(split_f), "r") as f:
            contents = [x.strip() for x in f.readlines()]

        self.images = [os.path.join(image_dir, x[:x.find(' ')] + '.jpg') for x in contents]
        if not self.is_test:
            self.masks = [os.path.join(mask_dir, x[:x.find(' ')] + '.png') for x in contents]
            self.lane_existences = [list(map(int, x[x.find(' '):].split())) for x in contents]
