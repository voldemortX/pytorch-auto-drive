# Mostly copied and modified from torch/vision/references/segmentation to support unlabeled data
# Copied functions from fmassa/vision-1 to support multi-dimensional masks loaded from numpy ndarray
# Update: The current torchvision github repo now supports tensor operation for all common transformations,
# you are encouraged to check it out
# Processing in (w, h), while providing public functions in (h, w)
#######################
# For transforms with multiple targets (masks, keypoints), target is formed as `dict{'padding_mask', 'keypoints', etc.}`
import numpy as np
from PIL import Image
from collections.abc import Sequence
import numbers
import random
import torch
from . import functional as F
from . import functional_keypoints as F_kp


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


class RandomApply(object):
    def __init__(self, transforms, apply_prob=0.5):
        self.transforms = transforms
        self.apply_prob = apply_prob

    def __call__(self, image, target):
        t = random.random()
        if t < self.apply_prob:
            for t in self.transforms:
                image, target = t(image, target)

        return image, target


class Resize(object):
    def __init__(self, size_image, size_label, ignore_x=-2):
        self.size_image = size_image
        self.size_label = size_label
        self.ignore_x = ignore_x

    @staticmethod
    def parse_resize(image, target, size_image, size_label, ori_size, ignore_x):
        image = F.resize(image, size_image, interpolation=Image.LINEAR)
        if isinstance(target, str):
            return image, target
        elif isinstance(target, dict):  # To keep BC
            if 'keypoints' in target.keys():
                target['keypoints'] = F_kp.resize(target['keypoints'], ori_size, size_label, ignore_x)
            # if 'padding_mask' in target.keys():
            #     target['padding_mask'] = F.resize(target['padding_mask'], size_label, interpolation=Image.NEAREST)
        else:
            target = F.resize(target, size_label, interpolation=Image.NEAREST)

        return image, target

    def __call__(self, image, target):
        w_ori, h_ori = F._get_image_size(image)

        return self.parse_resize(image, target, self.size_image, self.size_label, (h_ori, w_ori), self.ignore_x)


# Crop from up-left corner
class Crop(object):
    def __init__(self, size, ignore_x=-2):
        self.h, self.w = size
        self.ignore_x = ignore_x

    @staticmethod
    def parse_crop(image, target, top, left, height, width, ignore_x):
        # Crop with 4 degrees of freedom (top, left, height, width)
        image = F.crop(image, top, left, height, width)
        if isinstance(target, str):
            return image, target
        elif isinstance(target, dict):  # To keep BC
            if 'keypoints' in target.keys():
                target['keypoints'] = F_kp.crop(target['keypoints'], top, left, height, width, ignore_x)
            # if 'padding_mask' in target.keys():
            #     target['padding_mask'] = F.crop(target['padding_mask'], top, left, height, width)
        else:
            target = F.crop(target, top, left, height, width)

        return image, target

    def __call__(self, image, target):
        return self.parse_crop(image, target, 0, 0, self.h, self.w, self.ignore_x)


# Pad image with zeros, yet pad target with 255 (ignore label) on bottom & right if
# given a bigger desired size (or else nothing is done at all)
class ZeroPad(object):
    def __init__(self, size):
        self.h, self.w = size

    @staticmethod
    def zero_pad(image, target, h, w):
        ow, oh = F._get_image_size(image)
        pad_h = h - oh if oh < h else 0
        pad_w = w - ow if ow < w else 0
        image = F.pad(image, [0, 0, pad_w, pad_h], fill=0)
        if isinstance(target, str):
            return image, target
        elif isinstance(target, dict):  # To keep BC
            # Conveniently, since padding is on right & bottom, nothing needs to be done for keypoints
            if 'padding_mask' in target.keys():
                target['padding_mask'] = F.pad(target['padding_mask'], [0, 0, pad_w, pad_h], fill=1)
        else:
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
        tw, th = F._get_image_size(image)
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
    def __init__(self, min_size, max_size=None, ignore_x=-2):
        self.min_size = min_size
        if max_size is None:
            max_size = min_size
        self.max_size = max_size
        self.ignore_x = ignore_x

    def __call__(self, image, target):
        min_h, min_w = self.min_size
        max_h, max_w = self.max_size
        h = random.randint(min_h, max_h)
        w = random.randint(min_w, max_w)
        w_ori, h_ori = F._get_image_size(image)

        return Resize.parse_resize(image, target, [h, w], [h, w], (h_ori, w_ori), self.ignore_x)


class RandomScale(object):
    def __init__(self, min_scale, max_scale=None):
        self.min_scale = min_scale
        if max_scale is None:
            max_scale = min_scale
        self.max_scale = max_scale

    def __call__(self, image, target):
        scale = random.uniform(self.min_scale, self.max_scale)
        w, h = F._get_image_size(image)
        h = int(scale * h)
        w = int(scale * w)
        image = F.resize(image, [h, w], interpolation=Image.LINEAR)
        target = F.resize(target, [h, w], interpolation=Image.NEAREST)

        return image, target


class RandomCrop(object):
    def __init__(self, size, ignore_x=-2):
        self.size = size
        self.ignore_x = ignore_x

    @staticmethod
    def get_params(img, output_size):
        w, h = F._get_image_size(img)
        th, tw = output_size
        if w <= tw and h <= th:
            return 0, 0, h, w

        i = random.randint(0, h - th)
        j = random.randint(0, w - tw)
        return i, j, th, tw

    def __call__(self, image, target):
        # Pad if needed
        iw, ih = F._get_image_size(image)
        if ih < self.size[0] or iw < self.size[1]:
            # print(image.size())
            # print(self.size)
            image, target = ZeroPad.zero_pad(image, target,
                                             max(self.size[0], ih),
                                             max(self.size[1], iw))
        i, j, h, w = self.get_params(image, self.size)

        return Crop.parse_crop(image, target, i, j, h, w, self.ignore_x)


class RandomHorizontalFlip(object):
    def __init__(self, flip_prob, ignore_x):
        self.flip_prob = flip_prob
        self.ignore_x = -2

    def __call__(self, image, target):
        t = random.random()
        if t < self.flip_prob:
            image = F.hflip(image)
            if isinstance(target, str):
                return image, target
            elif isinstance(target, dict):  # To keep BC
                if 'keypoints' in target.keys():
                    target['keypoints'] = F_kp.hflip(target['keypoints'],
                                                     F._get_image_size(image)[0] / 2, self.ignore_x)
                # if 'padding_mask' in target.keys():
                #     target['padding_mask'] = F.hflip(target['padding_mask'])
            else:
                target = F.hflip(target)
        else:
            return image, target

        return image, target


class ToTensor(object):
    def __init__(self, keep_scale=False, reverse_channels=False):
        # keep_scale = True => Images or whatever are not divided by 255
        # reverse_channels = True => RGB images are changed to BGR (the default behavior of openCV & Caffe,
        #                                                           let's wish them all go to heaven,
        #                                                           for they wasted me days!)
        self.keep_scale = keep_scale
        self.reverse_channels = reverse_channels

    def __call__(self, image, target):
        image = self._pil_to_tensor(image)
        target = self.label_to_tensor(target)

        return image, target

    @staticmethod
    def label_to_tensor(pic):  # segmentation masks or keypoint arrays
        if isinstance(pic, str):
            return pic
        elif isinstance(pic, dict):
            if 'keypoints' in pic:
                pic['keypoints'] = torch.as_tensor(pic['keypoints'].copy(), dtype=torch.float32)
            if 'padding_mask' in pic:
                pic['padding_mask'] = torch.as_tensor(np.asarray(pic['padding_mask']).copy(), dtype=torch.uint8)
            return pic
        else:
            return torch.as_tensor(np.asarray(pic).copy(), dtype=torch.int64)

    def _pil_to_tensor(self, pic):
        # Convert a PIL Image to tensor (a direct copy)
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
    def __init__(self, mean, std, normalize_target=False, ignore_x=-2):
        self.mean = mean
        self.std = std
        self.normalize_target = normalize_target
        self.ignore_x = ignore_x

    def __call__(self, image, target):
        image = F.normalize(image, mean=self.mean, std=self.std)
        if self.normalize_target and not isinstance(target, str):
            if isinstance(target, dict):
                if 'keypoints' in target.keys():
                    w, h = F._get_image_size(image)
                    target['keypoints'] = F_kp.normalize(target['keypoints'], h, w, self.ignore_x)

        return image, target


# Init with a python list as the map (mainly for cityscapes's id -> train_id)
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
        wi, hi = F._get_image_size(image)
        wl, hl = F._get_image_size(target)
        if hi == hl and wi == wl:
            return image, target

        if self.l2i:
            target = F.resize(target, [hi, wi], interpolation=Image.NEAREST)
        else:
            image = F.resize(image, [hl, wl], interpolation=Image.LINEAR)

        return image, target


# TODO: Support fill color 255 for tensor inputs (supported in torchvision >= 0.9.0)
# Now fill color is fixed to 0 (background for lane detection label)
class RandomRotation(object):
    def __init__(self, degrees, expand=False, center=None, fill=None, ignore_x=-2):
        self.degrees = _setup_angle(degrees, name="degrees", req_sizes=(2, ))

        if center is not None:
            _check_sequence_input(center, "center", req_sizes=(2, ))

        self.center = center
        self.expand = expand
        self.fill = fill
        self.ignore_x = ignore_x

    @staticmethod
    def get_params(degrees):

        return random.uniform(degrees[0], degrees[1])

    def __call__(self, image, target):
        angle = self.get_params(self.degrees)
        image = F.rotate(image, angle, resample=Image.LINEAR, expand=self.expand, center=self.center, fill=0)
        if isinstance(target, dict):  # To keep BC
            if 'keypoints' in target.keys():
                w, h = F._get_image_size(image)
                target['keypoints'] = F_kp.rotate(target['keypoints'], angle, h, w, self.ignore_x)
            # if 'padding_mask' in target.keys():
            #     target['padding_mask'] = F.rotate(target['padding_mask'], angle, resample=Image.NEAREST,
            #                                       expand=self.expand, center=self.center, fill=1)
        else:
            target = F.rotate(target, angle, resample=Image.NEAREST, expand=self.expand, center=self.center, fill=255)

        return image, target


# TODO: Change to random
class ColorJitter(object):
    """Randomly change the brightness, contrast, saturation and hue of an image.
    If the image is torch Tensor, it is expected
    to have [..., 3, H, W] shape, where ... means an arbitrary number of leading dimensions.
    If img is PIL Image, mode "1", "L", "I", "F" and modes with transparency (alpha channel) are not supported.
    Args:
        brightness (float or tuple of float (min, max)): How much to jitter brightness.
            brightness_factor is chosen uniformly from [max(0, 1 - brightness), 1 + brightness]
            or the given [min, max]. Should be non negative numbers.
        contrast (float or tuple of float (min, max)): How much to jitter contrast.
            contrast_factor is chosen uniformly from [max(0, 1 - contrast), 1 + contrast]
            or the given [min, max]. Should be non negative numbers.
        saturation (float or tuple of float (min, max)): How much to jitter saturation.
            saturation_factor is chosen uniformly from [max(0, 1 - saturation), 1 + saturation]
            or the given [min, max]. Should be non negative numbers.
        hue (float or tuple of float (min, max)): How much to jitter hue.
            hue_factor is chosen uniformly from [-hue, hue] or the given [min, max].
            Should have 0<= hue <= 0.5 or -0.5 <= min <= max <= 0.5.
    """

    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0):
        super().__init__()
        self.brightness = self._check_input(brightness, 'brightness')
        self.contrast = self._check_input(contrast, 'contrast')
        self.saturation = self._check_input(saturation, 'saturation')
        self.hue = self._check_input(hue, 'hue', center=0, bound=(-0.5, 0.5),
                                     clip_first_on_zero=False)

    @torch.jit.unused
    def _check_input(self, value, name, center=1, bound=(0, float('inf')), clip_first_on_zero=True):
        if isinstance(value, numbers.Number):
            if value < 0:
                raise ValueError("If {} is a single number, it must be non negative.".format(name))
            value = [center - float(value), center + float(value)]
            if clip_first_on_zero:
                value[0] = max(value[0], 0.0)
        elif isinstance(value, (tuple, list)) and len(value) == 2:
            if not bound[0] <= value[0] <= value[1] <= bound[1]:
                raise ValueError("{} values should be between {}".format(name, bound))
        else:
            raise TypeError("{} should be a single number or a list/tuple with length 2.".format(name))

        # if value is 0 or (1., 1.) for brightness/contrast/saturation
        # or (0., 0.) for hue, do nothing
        if value[0] == value[1] == center:
            value = None
        return value

    @staticmethod
    def get_params(brightness, contrast, saturation, hue):
        """Get the parameters for the randomized transform to be applied on image.
        Args:
            brightness (tuple of float (min, max), optional): The range from which the brightness_factor is chosen
                uniformly. Pass None to turn off the transformation.
            contrast (tuple of float (min, max), optional): The range from which the contrast_factor is chosen
                uniformly. Pass None to turn off the transformation.
            saturation (tuple of float (min, max), optional): The range from which the saturation_factor is chosen
                uniformly. Pass None to turn off the transformation.
            hue (tuple of float (min, max), optional): The range from which the hue_factor is chosen uniformly.
                Pass None to turn off the transformation.
        Returns:
            tuple: The parameters used to apply the randomized transform
            along with their random order.
        """
        fn_idx = torch.randperm(4)

        b = None if brightness is None else float(torch.empty(1).uniform_(brightness[0], brightness[1]))
        c = None if contrast is None else float(torch.empty(1).uniform_(contrast[0], contrast[1]))
        s = None if saturation is None else float(torch.empty(1).uniform_(saturation[0], saturation[1]))
        h = None if hue is None else float(torch.empty(1).uniform_(hue[0], hue[1]))

        return fn_idx, b, c, s, h

    def __call__(self, image, target):
        """
        Args:
            image (PIL Image or Tensor): Input image.
        Returns:
            PIL Image or Tensor: Color jittered image.
        """
        fn_idx, brightness_factor, contrast_factor, saturation_factor, hue_factor = \
            self.get_params(self.brightness, self.contrast, self.saturation, self.hue)

        for fn_id in fn_idx:
            if fn_id == 0 and brightness_factor is not None:
                image = F.adjust_brightness(image, brightness_factor)
            elif fn_id == 1 and contrast_factor is not None:
                image = F.adjust_contrast(image, contrast_factor)
            elif fn_id == 2 and saturation_factor is not None:
                image = F.adjust_saturation(image, saturation_factor)
            elif fn_id == 3 and hue_factor is not None:
                image = F.adjust_hue(image, hue_factor)

        return image, target


# Lighting adjustment with eigen vectors from LSTR:
# https://github.com/liuruijin17/LSTR/blob/6044f7b2c5892dba7201c273ee632b4962350223/utils/image.py#L12
class RandomLighting(object):
    def __init__(self, mean, std, eigen_value, eigen_vector):
        self.mean = mean
        self.std = std
        self.eigen_value = torch.tensor(eigen_value, dtype=torch.float32)
        self.eigen_vector = torch.tensor(eigen_vector, dtype=torch.float32)

    def __call__(self, image, target):
        alpha = torch.normal(self.mean, self.std, (3, ), dtype=torch.float32)

        return F.adjust_lighting(image, alpha, self.eigen_value, self.eigen_vector), target
