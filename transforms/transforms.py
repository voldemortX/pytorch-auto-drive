# Mostly copied and modified from torch/vision/references/segmentation to support unlabeled data
# Copied functions from fmassa/vision-1 to support multi-dimensional masks loaded from numpy ndarray
# Update: The current torchvision github repo now supports tensor operation for all common transformations,
# you are encouraged to check it out
import numpy as np
from PIL import Image
from collections.abc import Sequence
import numbers
import random
import torch
import math
from . import functional as F


# For 2/3 dimensional tensors only
def get_tensor_image_size(img):
    if img.dim() == 2:
        h, w = img.size()
    else:
        h = img.size()[1]
        w = img.size()[2]

    return h, w


def _check_sequence_input(x, name, req_sizes):
    msg = req_sizes[0] if len(req_sizes) < 2 else " or ".join([str(s) for s in req_sizes])
    if not isinstance(x, Sequence):
        raise TypeError("{} should be a sequence of length {}.".format(name, msg))
    if len(x) not in req_sizes:
        raise ValueError("{} should be sequence of length {}.".format(name, msg))


def _setup_angle(x, name, req_sizes=(2, )):
    if isinstance(x, numbers.Number):
        if x < 0:
            raise ValueError("If {} is a single number, it must be positive.".format(name))
        x = [-x, x]
    else:
        _check_sequence_input(x, name, req_sizes)

    return [float(d) for d in x]


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)

        return image, target


class Resize(object):
    def __init__(self, size_image, size_label):
        self.size_image = size_image
        self.size_label = size_label

    @staticmethod
    def transform_points(points, in_size, out_size):
        # Resize a np.array (N x 2) of points (x, y), original axis start from top-left corner
        in_h, in_w = in_size
        out_h, out_w = out_size
        scale = np.array([out_h / in_h, out_w / in_w])
        return points * scale

    def __call__(self, image, target):
        image = F.resize(image, self.size_image, interpolation=Image.LINEAR)
        if isinstance(target, str):
            return image, target
        elif isinstance(target, np.ndarray):
            in_size = F._get_image_size(image)
            target = self.transform_points(target, in_size, self.size_label)
        else:
            target = F.resize(target, self.size_label, interpolation=Image.NEAREST)

        return image, target


# Crop from up-left corner
class Crop(object):
    def __init__(self, size):
        self.h, self.w = size

    def __call__(self, image, target):
        image = F.crop(image, 0, 0, self.h, self.w)
        target = F.crop(target, 0, 0, self.h, self.w)

        return image, target


# Pad image with zeros, yet pad target with 255 (ignore label) on bottom & right if
# given a bigger desired size (or else nothing is done at all)
class ZeroPad(object):
    def __init__(self, size):
        self.h, self.w = size

    @staticmethod
    def zero_pad(image, target, h, w):
        oh, ow = get_tensor_image_size(target)
        pad_h = h - oh if oh < h else 0
        pad_w = w - ow if ow < w else 0
        image = F.pad(image, [0, 0, pad_w, pad_h], fill=0)
        target = F.pad(target, [0, 0, pad_w, pad_h], fill=255)

        return image, target

    def __call__(self, image, target):
        return self.zero_pad(image, target, self.h, self.w)


# Random translation in pixels
# Random translation = Zero pad + Random crop
class RandomTranslation(object):
    def __init__(self, trans_h, trans_w):
        self.trans_h = trans_h
        self.trans_w = trans_w

    def __call__(self, image, target):
        th, tw = get_tensor_image_size(image)
        image = F.pad(image, [self.trans_w, self.trans_h, self.trans_w, self.trans_h], fill=0)
        target = F.pad(target, [self.trans_w, self.trans_h, self.trans_w, self.trans_h], fill=255)
        i, j, h, w = RandomCrop.get_params(image, (th, tw))
        image = F.crop(image, i, j, h, w)
        target = F.crop(target, i, j, h, w)

        return image, target


class RandomZeroPad(object):
    def __init__(self, pad_h, pad_w):
        self.pad_h = pad_h
        self.pad_w = pad_w

    def __call__(self, image, target):
        r = random.randint(-self.pad_w, self.pad_w)
        b = random.randint(-self.pad_h, self.pad_h)
        l = 0
        t = 0
        if r < 0:
            l = -r
            r = 0
        if b < 0:
            t = -b
            b = 0

        image = F.pad(image, [l, t, r, b], fill=0)
        target = F.pad(target, [l, t, r, b], fill=255)

        return image, target


class RandomResize(object):
    def __init__(self, min_size, max_size=None):
        self.min_size = min_size
        if max_size is None:
            max_size = min_size
        self.max_size = max_size

    def __call__(self, image, target):
        min_h, min_w = self.min_size
        max_h, max_w = self.max_size
        h = random.randint(min_h, max_h)
        w = random.randint(min_w, max_w)
        image = F.resize(image, [h, w], interpolation=Image.LINEAR)
        if isinstance(target, str):
            return image, target
        elif isinstance(target, np.ndarray):
            in_size = F._get_image_size(image)
            target = Resize.transform_points(target, in_size, (h, w))
        else:
            target = F.resize(target, [h, w], interpolation=Image.NEAREST)

        return image, target


class RandomScale(object):
    def __init__(self, min_scale, max_scale=None):
        self.min_scale = min_scale
        if max_scale is None:
            max_scale = min_scale
        self.max_scale = max_scale

    def __call__(self, image, target):
        scale = random.uniform(self.min_scale, self.max_scale)
        h, w = get_tensor_image_size(image)
        h = int(scale * h)
        w = int(scale * w)
        image = F.resize(image, [h, w], interpolation=Image.LINEAR)
        target = F.resize(target, [h, w], interpolation=Image.NEAREST)

        return image, target


class RandomCrop(object):
    def __init__(self, size):
        self.size = size

    @staticmethod
    def get_params(img, output_size):
        h, w = get_tensor_image_size(img)
        th, tw = output_size
        if w <= tw and h <= th:
            return 0, 0, h, w

        i = random.randint(0, h - th)
        j = random.randint(0, w - tw)
        return i, j, th, tw

    def __call__(self, image, target):
        # Pad if needed
        ih, iw = get_tensor_image_size(image)
        if ih < self.size[0] or iw < self.size[1]:
            # print(image.size())
            # print(self.size)
            image, target = ZeroPad.zero_pad(image, target,
                                             max(self.size[0], ih),
                                             max(self.size[1], iw))
        i, j, h, w = self.get_params(image, self.size)
        image = F.crop(image, i, j, h, w)
        target = F.crop(target, i, j, h, w)

        return image, target


class RandomHorizontalFlip(object):
    def __init__(self, flip_prob):
        self.flip_prob = flip_prob

    def __call__(self, image, target):
        t = random.random()
        if t < self.flip_prob:
            image = F.hflip(image)
        target = target if (isinstance(target, str) or t >= self.flip_prob) else F.hflip(target)

        return image, target


class ToTensor(object):
    def __init__(self, keep_scale=False, reverse_channels=False):
        # keep_scale = True => Images or whatever are not divided by 255
        # reverse_channels = True => RGB images are changed to BGR(the default behavior of openCV & Caffe,
        #                                                          let's wish them all go to heaven,
        #                                                          for they wasted me days!)
        self.keep_scale = keep_scale
        self.reverse_channels = reverse_channels

    def __call__(self, image, target):
        image = self._pil_to_tensor(image)
        target = self.label_to_tensor(target)

        return image, target

    @staticmethod
    def label_to_tensor(pic):  # 3 dimensional arrays or normal segmentation masks
        if isinstance(pic, np.ndarray):
            return torch.as_tensor(pic.transpose((2, 0, 1)), dtype=torch.float32)
        elif isinstance(pic, str):
            return pic
        else:
            return torch.as_tensor(np.asarray(pic).copy(), dtype=torch.int64)

    def _pil_to_tensor(self, pic):
        # Convert a PIL Image to tensor(a direct copy)
        if pic.mode == 'I':
            img = torch.from_numpy(np.array(pic, np.int32, copy=False))
        elif pic.mode == 'I;16':
            img = torch.from_numpy(np.array(pic, np.int16, copy=False))
        elif pic.mode == 'F':
            img = torch.from_numpy(np.array(pic, np.float32, copy=False))
        elif pic.mode == '1':
            img = 255 * torch.from_numpy(np.array(pic, np.uint8, copy=False))
        else:
            img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
        # PIL image mode: L, LA, P, I, F, RGB, YCbCr, RGBA, CMYK
        if pic.mode == 'YCbCr':
            nchannel = 3
        elif pic.mode == 'I;16':
            nchannel = 1
        else:
            nchannel = len(pic.mode)
        img = img.view(pic.size[1], pic.size[0], nchannel)
        if self.reverse_channels:  # Beware this only works with 3 channels(can't use -1 with tensors)
            img = img[:, :, [2, 1, 0]]
        # put it from HWC to CHW format
        # yikes, this transpose takes 80% of the loading time/CPU
        img = img.transpose(0, 1).transpose(0, 2).contiguous()
        if isinstance(img, torch.ByteTensor):
            if self.keep_scale:
                return img.float()
            else:
                return img.float().div(255)
        else:
            return img


class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image, target):
        image = F.normalize(image, mean=self.mean, std=self.std)

        return image, target


# Init with a python list as the map(mainly for cityscapes's id -> train_id)
class LabelMap(object):
    def __init__(self, label_id_map, outlier=False):
        self.label_id_map = torch.tensor(label_id_map)
        self.outlier = outlier

    def __call__(self, image, target):
        if self.outlier:
            target[target >= self.label_id_map.shape[0]] = 0  # Label 0 is usually ignored
        target = self.label_id_map[target]

        return image, target


# Match label and image size
class MatchSize(object):
    def __init__(self, l2i=True):
        self.l2i = l2i  # Match (l)abel to (i)mage

    def __call__(self, image, target):
        hi, wi = get_tensor_image_size(image)
        hl, wl = get_tensor_image_size(target)
        if hi == hl and wi == wl:
            return image, target

        if self.l2i:
            target = F.resize(target, [hi, wi], interpolation=Image.NEAREST)
        else:
            image = F.resize(image, [hl, wl], interpolation=Image.LINEAR)

        return image, target


# TODO: Support fill color 255 for tensor inputs (supported in torchvision nightly)
# Now fill color is fixed to 0 (background for lane detection label)
class RandomRotation(object):
    def __init__(self, degrees, expand=False, center=None, fill=None):
        self.degrees = _setup_angle(degrees, name="degrees", req_sizes=(2, ))

        if center is not None:
            _check_sequence_input(center, "center", req_sizes=(2, ))

        self.center = center
        self.expand = expand
        self.fill = fill

    @staticmethod
    def get_params(degrees):

        return random.uniform(degrees[0], degrees[1])

    @staticmethod
    def transform_points(points, angle, h, w):
        # Rotate a np.array (N x 2) of points (x, y) anti-clockwise, original axis start from top-left corner
        # TODO: Mask -2
        offset = np.array([h / 2, w / 2])
        matrix = np.array([[math.cos(angle / 360.0 * math.pi), math.sin(angle / 360.0 * math.pi)],
                           [math.sin(-angle / 360.0 * math.pi), math.cos(angle / 360.0 * math.pi)]])
        points = np.matmul((points - offset), matrix) + offset
        indices = (points[:, 0] < h) + (points[:, 1] < w) + (points > 0).sum(axis=1, dtype=np.bool)

        return points[indices]

    def __call__(self, image, target):
        angle = self.get_params(self.degrees)
        image = F.rotate(image, angle, resample=Image.LINEAR, expand=self.expand, center=self.center, fill=0)
        if isinstance(target, np.ndarray):
            h, w = F._get_image_size(image)
            target = self.transform_points(target, angle, h, w)
        else:
            target = F.rotate(target, angle, resample=Image.NEAREST, expand=self.expand, center=self.center, fill=255)

        return image, target
