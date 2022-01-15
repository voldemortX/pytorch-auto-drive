import os
import torch
import torchvision
import json
import numpy as np
from PIL import Image

from .builder import DATASETS
from ..curve_utils import BezierSampler, get_valid_points


class _BezierLaneDataset(torchvision.datasets.VisionDataset):
    # BezierLaneNet dataset, includes binary seg labels
    keypoint_color = [0, 0, 0]

    def __init__(self, root, image_set='train', transforms=None, transform=None, target_transform=None,
                 order=3, num_sample_points=100, aux_segmentation=False):
        super().__init__(root, transforms, transform, target_transform)
        self.aux_segmentation = aux_segmentation
        self.bezier_sampler = BezierSampler(order=order, num_sample_points=num_sample_points)
        if image_set == 'valfast':
            raise NotImplementedError('valfast Not supported yet!')
        elif image_set == 'test' or image_set == 'val':  # Different format (without lane existence annotations)
            self.test = 2
        elif image_set == 'val_train':
            self.test = 3
        else:
            self.test = 0

        self.init_dataset()

        if image_set != 'valfast':
            self.bezier_labels = os.path.join(self.bezier_labels_dir, image_set + '_' + str(order) + '.json')
        elif image_set == 'valfast':
            raise ValueError

        self.image_set = image_set
        self.splits_dir = os.path.join(root, 'lists')
        self._init_all()

    def init_dataset(self, root):
        raise NotImplementedError

    def __getitem__(self, index):
        # Return x (input image) & y (mask image, i.e. pixel-wise supervision) & lane existence (a list),
        # if not just testing,
        # else just return input image.
        img = Image.open(self.images[index]).convert('RGB')
        if self.test >= 2:
            target = self.masks[index]
        else:
            if self.aux_segmentation:
                target = {'keypoints': self.beziers[index],
                          'segmentation_mask': Image.open(self.masks[index])}
            else:
                target = {'keypoints': self.beziers[index]}

        # Transforms
        if self.transforms is not None:
            img, target = self.transforms(img, target)

        if self.test == 0:
            target = self._post_process(target)

        return img, target

    def __len__(self):
        return len(self.images)

    def loader_bezier(self):
        results = []
        with open(self.bezier_labels, 'r') as f:
            results += [json.loads(x.strip()) for x in f.readlines()]
        beziers = []
        for lanes in results:
            temp_lane = []
            for lane in lanes['bezier_control_points']:
                temp_cps = []
                for i in range(0, len(lane), 2):
                    temp_cps.append([lane[i], lane[i + 1]])
                temp_lane.append(temp_cps)
            beziers.append(np.array(temp_lane, dtype=np.float32))
        return beziers

    def _init_all(self):
        # Got the lists from 4 datasets to be in the same format
        data_list = 'train.txt' if self.image_set == 'val_train' else self.image_set + '.txt'
        split_f = os.path.join(self.splits_dir, data_list)
        with open(split_f, "r") as f:
            contents = [x.strip() for x in f.readlines()]
        if self.test == 2:  # Test
            self.images = [os.path.join(self.image_dir, x + self.image_suffix) for x in contents]
            self.masks = [os.path.join(self.output_prefix, x + self.output_suffix) for x in contents]
        elif self.test == 3:  # Test
            self.images = [os.path.join(self.image_dir, x[:x.find(' ')] + self.image_suffix) for x in contents]
            self.masks = [os.path.join(self.output_prefix, x[:x.find(' ')] + self.output_suffix) for x in contents]
        elif self.test == 1:  # Val
            self.images = [os.path.join(self.image_dir, x[:x.find(' ')] + self.image_suffix) for x in contents]
            self.masks = [os.path.join(self.mask_dir, x[:x.find(' ')] + '.png') for x in contents]
        else:  # Train
            self.images = [os.path.join(self.image_dir, x[:x.find(' ')] + self.image_suffix) for x in contents]
            if self.aux_segmentation:
                self.masks = [os.path.join(self.mask_dir, x[:x.find(' ')] + '.png') for x in contents]
            self.beziers = self.loader_bezier()

    def _post_process(self, target, ignore_seg_index=255):
        # Get sample points and delete invalid lines (< 2 points)
        if target['keypoints'].numel() != 0:  # No-lane cases can be handled in loss computation
            sample_points = self.bezier_sampler.get_sample_points(target['keypoints'])
            valid_lanes = get_valid_points(sample_points).sum(dim=-1) >= 2
            target['keypoints'] = target['keypoints'][valid_lanes]
            target['sample_points'] = sample_points[valid_lanes]
        else:
            target['sample_points'] = torch.tensor([], dtype=target['keypoints'].dtype)

        if 'segmentation_mask' in target.keys():  # Map to binary (0 1 255)
            positive_mask = (target['segmentation_mask'] > 0) * (target['segmentation_mask'] != ignore_seg_index)
            target['segmentation_mask'][positive_mask] = 1

        return target


# TuSimple
@DATASETS.register()
class TuSimpleAsBezier(_BezierLaneDataset):
    def init_dataset(self, root):
        self.image_dir = os.path.join(root, 'clips')
        self.bezier_labels_dir = os.path.join(root, 'bezier_labels')
        self.mask_dir = os.path.join(root, 'segGT6')
        self.output_prefix = 'clips'
        self.output_suffix = '.jpg'
        self.image_suffix = '.jpg'


# CULane
@DATASETS.register()
class CULaneAsBezier(_BezierLaneDataset):
    def init_dataset(self, root):
        self.image_dir = root
        self.bezier_labels_dir = os.path.join(root, 'bezier_labels')
        self.mask_dir = os.path.join(root, 'laneseg_label_w16')
        self.output_prefix = './output'
        self.output_suffix = '.lines.txt'
        self.image_suffix = '.jpg'
        if not os.path.exists(self.output_prefix):
            os.makedirs(self.output_prefix)


# LLAMAS
@DATASETS.register()
class LLAMAS_AsBezier(_BezierLaneDataset):
    def init_dataset(self, root):
        self.image_dir = os.path.join(root, 'color_images')
        self.bezier_labels_dir = os.path.join(root, 'bezier_labels')
        self.mask_dir = os.path.join(root, 'laneseg_labels')
        self.output_prefix = './output'
        self.output_suffix = '.lines.txt'
        self.image_suffix = '.png'
        if not os.path.exists(self.output_prefix):
            os.makedirs(self.output_prefix)


# Curvelanes
@DATASETS.register()
class Curvelanes_AsBezier(CULaneAsBezier):
    # TODO: Match formats
    def _init_all(self):
        # Got the lists from 4 datasets to be in the same format
        data_list = 'train.txt' if self.image_set == 'val_train' else self.image_set + '.txt'
        split_f = os.path.join(self.splits_dir, data_list)
        with open(split_f, "r") as f:
            contents = [x.strip() for x in f.readlines()]
        if self.test == 2:  # Test
            self.images = [os.path.join(self.image_dir, x + self.image_suffix) for x in contents]
            self.masks = [os.path.join(self.output_prefix, x + self.output_suffix) for x in contents]
        elif self.test == 3:  # Test
            self.images = [os.path.join(self.image_dir, x[:x.find(' ')] + self.image_suffix) for x in contents]
            self.masks = [os.path.join(self.output_prefix, x[:x.find(' ')] + self.output_suffix) for x in contents]
        elif self.test == 1:  # Val
            self.images = [os.path.join(self.image_dir, x[:x.find(' ')] + self.image_suffix) for x in contents]
            self.masks = [os.path.join(self.mask_dir, x[:x.find(' ')] + '.png') for x in contents]
        else:  # Train
            self.images = [os.path.join(self.image_dir, x + self.image_suffix) for x in contents]
            if self.aux_segmentation:
                self.masks = [os.path.join(self.mask_dir, x[:x.find(' ')] + '.png') for x in contents]
            self.beziers = self.loader_bezier()
