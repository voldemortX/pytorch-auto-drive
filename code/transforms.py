# Mostly copied and modified from torch/vision/references/segmentation to support unlabeled data
# Copied functions from fmassa/vision-1 to support multi-dimensional masks loaded from numpy ndarray
import numpy as np
from PIL import Image
import random
import torch
import functional as F


# For 2/3 dimensional tensors only
def get_tensor_image_size(img):
    if img.dim() == 2:
        h, w = img.size()
    else:
        h = img.size()[1]
        w = img.size()[2]

    return h, w


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

    def __call__(self, image, target):
        image = F.resize(image, self.size_image, interpolation=Image.LINEAR)
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
        image = F.pad(image, (0, 0, pad_w, pad_h), fill=0)
        target = F.pad(target, (0, 0, pad_w, pad_h), fill=255)

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
        image = F.pad(image, (self.trans_w, self.trans_h, self.trans_w, self.trans_h), fill=0)
        target = F.pad(target, (self.trans_w, self.trans_h, self.trans_w, self.trans_h), fill=255)
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

        image = F.pad(image, (l, t, r, b), fill=0)
        target = F.pad(target, (l, t, r, b), fill=255)

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
        image = F.resize(image, (h, w), interpolation=Image.LINEAR)
        target = F.resize(target, (h, w), interpolation=Image.NEAREST)

        return image, target


class RandomCrop(object):
    def __init__(self, size):
        self.size = size

    @staticmethod
    def get_params(img, output_size):
        h, w = get_tensor_image_size(img)
        th, tw = output_size
        if w == tw and h == th:
            return 0, 0, h, w

        i = random.randint(0, h - th)
        j = random.randint(0, w - tw)
        return i, j, th, tw

    def __call__(self, image, target):
        # Pad if needed
        ih, iw = get_tensor_image_size(image)
        if ih < self.size[0] or iw < self.size[1]:
            print(image.size())
            print(self.size)
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
        target = target if (type(target) == str or t >= self.flip_prob) else F.hflip(target)

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
            target[target > self.label_id_map.shape[0]] = 0  # Label 0 is usually ignored
        target = self.label_id_map[target]

        return image, target
