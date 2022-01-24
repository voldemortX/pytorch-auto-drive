# Mostly copied and modified from torch/vision/references/segmentation to support unlabeled data,
# while adding registers and modified compose transforms to build from dicts.
# Copied functions from fmassa/vision-1 to support multi-dimensional masks loaded from numpy ndarray
# Update: The current torchvision github repo now supports tensor operation for all common transformations,
# you are encouraged to check it out
# Processing in (w, h), while providing public functions in (h, w)
#######################
# For transforms with multiple targets (masks, keypoints), target is formed as `dict{'padding_mask', 'keypoints', etc.}`
import numpy as np
import numbers
import random
import torch
from collections.abc import Sequence
from PIL import Image
from scipy.interpolate import InterpolatedUnivariateSpline

from . import functional as F
from . import functional_keypoints as F_kp
from .builder import TRANSFORMS

__all__ = [
    'ColorJitter',
    'Compose',
    'Crop',
    'LabelMap',
    'MatchSize',
    'Normalize',
    'RandomAffine',
    'RandomApply',
    'RandomCrop',
    'RandomHorizontalFlip',
    'RandomLighting',
    'RandomResize',
    'RandomRotation',
    'RandomScale',
    'RandomTranslation',
    'RandomZeroPad',
    'Resize',
    'ToTensor',
    'ZeroPad',
    'ToXOffset'
]


def _check_sequence_input(x, name, req_sizes):
    msg = req_sizes[0] if len(req_sizes) < 2 else " or ".join([str(s) for s in req_sizes])
    if not isinstance(x, Sequence):
        raise TypeError("{} should be a sequence of length {}.".format(name, msg))
    if len(x) not in req_sizes:
        raise ValueError("{} should be sequence of length {}.".format(name, msg))


def _setup_angle(x, name, req_sizes=(2,)):
    if isinstance(x, numbers.Number):
        if x < 0:
            raise ValueError("If {} is a single number, it must be positive.".format(name))
        x = [-x, x]
    else:
        _check_sequence_input(x, name, req_sizes)

    return [float(d) for d in x]


@TRANSFORMS.register()
class Compose(object):
    def __init__(self, transforms):
        self.transforms = [t if callable(t) else TRANSFORMS.from_dict(t) for t in transforms]

    def __call__(self, image, target=None):
        for t in self.transforms:
            image, target = t(image, target)
        if target is None:
            return image
        else:
            return image, target


@TRANSFORMS.register()
class RandomApply(object):
    def __init__(self, transforms, apply_prob=0.5):
        self.transforms = [t if callable(t) else TRANSFORMS.from_dict(t) for t in transforms]
        self.apply_prob = apply_prob

    def __call__(self, image, target):
        t = random.random()
        if t < self.apply_prob:
            for t in self.transforms:
                image, target = t(image, target)

        return image, target


@TRANSFORMS.register()
class Resize(object):
    def __init__(self, size_image, size_label, ignore_x=-2):
        self.size_image = size_image
        self.size_label = size_label
        self.ignore_x = ignore_x

    @staticmethod
    def parse_resize(image, target, size_image, size_label, ori_size, ignore_x):
        image = F.resize(image, size_image, interpolation=Image.LINEAR)
        if target is None or isinstance(target, str):
            return image, target
        elif isinstance(target, dict):  # To keep BC
            if 'keypoints' in target.keys():
                target['keypoints'] = F_kp.resize(target['keypoints'], ori_size, size_label, ignore_x)
            # if 'padding_mask' in target.keys():
            #     target['padding_mask'] = F.resize(target['padding_mask'], size_label, interpolation=Image.NEAREST)
            if 'segmentation_mask' in target.keys():
                target['segmentation_mask'] = F.resize(target['segmentation_mask'],
                                                       size_label, interpolation=Image.NEAREST)
        else:
            target = F.resize(target, size_label, interpolation=Image.NEAREST)

        return image, target

    def __call__(self, image, target=None):
        w_ori, h_ori = F._get_image_size(image)

        return self.parse_resize(image, target, self.size_image, self.size_label, (h_ori, w_ori), self.ignore_x)


# Crop from up-left corner
@TRANSFORMS.register()
class Crop(object):
    def __init__(self, size, ignore_x=-2):
        self.h, self.w = size
        self.ignore_x = ignore_x

    @staticmethod
    def parse_crop(image, target, top, left, height, width, ignore_x):
        # Crop with 4 degrees of freedom (top, left, height, width)
        image = F.crop(image, top, left, height, width)
        if target is None or isinstance(target, str):
            return image, target
        elif isinstance(target, dict):  # To keep BC
            if 'keypoints' in target.keys():
                target['keypoints'] = F_kp.crop(target['keypoints'], top, left, height, width, ignore_x)
            # if 'padding_mask' in target.keys():
            #     target['padding_mask'] = F.crop(target['padding_mask'], top, left, height, width)
            if 'segmentation_mask' in target.keys():
                target['segmentation_mask'] = F.crop(target['segmentation_mask'], top, left, height, width)
        else:
            target = F.crop(target, top, left, height, width)

        return image, target

    def __call__(self, image, target=None):
        return self.parse_crop(image, target, 0, 0, self.h, self.w, self.ignore_x)


# Pad image with zeros, yet pad target with 255 (ignore label) on bottom & right if
# given a bigger desired size (or else nothing is done at all)
@TRANSFORMS.register()
class ZeroPad(object):
    def __init__(self, size):
        self.h, self.w = size

    @staticmethod
    def zero_pad(image, target, h, w):
        ow, oh = F._get_image_size(image)
        pad_h = h - oh if oh < h else 0
        pad_w = w - ow if ow < w else 0
        image = F.pad(image, [0, 0, pad_w, pad_h], fill=0)
        if target is None or isinstance(target, str):
            return image, target
        elif isinstance(target, dict):  # To keep BC
            # Conveniently, since padding is on right & bottom, nothing needs to be done for keypoints
            if 'padding_mask' in target.keys():
                target['padding_mask'] = F.pad(target['padding_mask'], [0, 0, pad_w, pad_h], fill=1)
            if 'segmentation_mask' in target.keys():
                target['segmentation_mask'] = F.pad(target['segmentation_mask'], [0, 0, pad_w, pad_h], fill=255)
        else:
            target = F.pad(target, [0, 0, pad_w, pad_h], fill=255)

        return image, target

    def __call__(self, image, target=None):
        return self.zero_pad(image, target, self.h, self.w)


# Random translation in pixels
# Random translation = Zero pad + Random crop
@TRANSFORMS.register()
class RandomTranslation(object):
    def __init__(self, trans_h, trans_w, ignore_x):
        self.trans_h = trans_h
        self.trans_w = trans_w
        self.ignore_x = ignore_x

    def __call__(self, image, target):
        tw, th = F._get_image_size(image)
        image = F.pad(image, [self.trans_w, self.trans_h, self.trans_w, self.trans_h], fill=0)
        i, j, h, w = RandomCrop.get_params(image, (th, tw))
        image = F.crop(image, i, j, h, w)
        if target is None or isinstance(target, str):
            return image, target
        elif isinstance(target, dict):  # To keep BC
            if 'keypoints' in target.keys():
                target['keypoints'] = F_kp.translate(target['keypoints'], tw - j, th - i, th, tw, self.ignore_x)
            # if 'padding_mask' in target.keys():
            #     pass
            if 'segmentation_mask' in target.keys():
                target['segmentation_mask'] = F.pad(target['segmentation_mask'],
                                                    [self.trans_w, self.trans_h, self.trans_w, self.trans_h], fill=255)
                target['segmentation_mask'] = F.crop(target['segmentation_mask'], i, j, h, w)
        else:
            target = F.pad(target, [self.trans_w, self.trans_h, self.trans_w, self.trans_h], fill=255)
            target = F.crop(target, i, j, h, w)

        return image, target


@TRANSFORMS.register()
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


@TRANSFORMS.register()
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


@TRANSFORMS.register()
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


@TRANSFORMS.register()
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


@TRANSFORMS.register()
class RandomHorizontalFlip(object):
    def __init__(self, flip_prob, ignore_x=-2):
        self.flip_prob = flip_prob
        self.ignore_x = ignore_x

    def __call__(self, image, target=None):
        t = random.random()
        if t < self.flip_prob:
            image = F.hflip(image)
            if target is None or isinstance(target, str):
                return image, target
            elif isinstance(target, dict):  # To keep BC
                if 'keypoints' in target.keys():
                    target['keypoints'] = F_kp.hflip(target['keypoints'],
                                                     F._get_image_size(image)[0] / 2, self.ignore_x)
                # if 'padding_mask' in target.keys():
                #     target['padding_mask'] = F.hflip(target['padding_mask'])
                if 'segmentation_mask' in target.keys():
                    target['segmentation_mask'] = F.hflip(target['segmentation_mask'])
            else:
                target = F.hflip(target)
        else:
            return image, target

        return image, target


@TRANSFORMS.register()
class ToTensor(object):
    def __init__(self, keep_scale=False, reverse_channels=False):
        # keep_scale = True => Images or whatever are not divided by 255
        # reverse_channels = True => RGB images are changed to BGR (the default behavior of openCV & Caffe,
        #                                                           let's wish them all go to heaven,
        #                                                           for they wasted me days!)
        self.keep_scale = keep_scale
        self.reverse_channels = reverse_channels

    def __call__(self, image, target=None):
        image = self._pil_to_tensor(image)
        target = self.label_to_tensor(target)

        return image, target

    @staticmethod
    def label_to_tensor(pic):  # segmentation masks or keypoint arrays
        if pic is None or isinstance(pic, str):
            return pic
        elif isinstance(pic, dict):
            if 'keypoints' in pic:
                pic['keypoints'] = torch.as_tensor(pic['keypoints'].copy(), dtype=torch.float32)
            if 'padding_mask' in pic:
                pic['padding_mask'] = torch.as_tensor(np.asarray(pic['padding_mask']).copy(), dtype=torch.uint8)
            if 'segmentation_mask' in pic:
                pic['segmentation_mask'] = torch.as_tensor(np.asarray(pic['segmentation_mask']).copy(),
                                                           dtype=torch.uint8)
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


@TRANSFORMS.register()
class Normalize(object):
    def __init__(self, mean, std, normalize_target=True, ignore_x=-2):
        # Default values changed (2021.12.18)
        self.mean = mean
        self.std = std
        self.normalize_target = normalize_target
        self.ignore_x = ignore_x

    def __call__(self, image, target=None):
        image = F.normalize(image, mean=self.mean, std=self.std)
        if self.normalize_target and not (target is None or isinstance(target, str)):
            if isinstance(target, dict):
                if 'keypoints' in target.keys():
                    w, h = F._get_image_size(image)
                    target['keypoints'] = F_kp.normalize(target['keypoints'], h, w, self.ignore_x)

        return image, target


# Init with a python list as the map (mainly for cityscapes's id -> train_id)
@TRANSFORMS.register()
class LabelMap(object):
    def __init__(self, label_id_map, outlier=False):
        self.label_id_map = torch.tensor(label_id_map)
        self.outlier = outlier

    def __call__(self, image, target=None):
        if target is None:
            return image, target
        if self.outlier:
            target[target >= self.label_id_map.shape[0]] = 0  # Label 0 is usually ignored
        target = self.label_id_map[target]

        return image, target


# Match label and image size
@TRANSFORMS.register()
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
@TRANSFORMS.register()
class RandomRotation(object):
    def __init__(self, degrees, expand=False, center=None, fill=None, ignore_x=-2):
        self.degrees = _setup_angle(degrees, name="degrees", req_sizes=(2,))

        if center is not None:
            _check_sequence_input(center, "center", req_sizes=(2,))

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
            if 'segmentation_mask' in target.keys():
                target['segmentation_mask'] = F.rotate(target['segmentation_mask'], angle, resample=Image.NEAREST,
                                                       expand=self.expand, center=self.center, fill=255)
        else:
            target = F.rotate(target, angle, resample=Image.NEAREST, expand=self.expand, center=self.center, fill=255)

        return image, target


# TODO: Change to random
@TRANSFORMS.register()
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
@TRANSFORMS.register()
class RandomLighting(object):
    def __init__(self, mean, std, eigen_value, eigen_vector):
        self.mean = mean
        self.std = std
        self.eigen_value = torch.tensor(eigen_value, dtype=torch.float32)
        self.eigen_vector = torch.tensor(eigen_vector, dtype=torch.float32)

    def __call__(self, image, target):
        alpha = torch.normal(self.mean, self.std, (3,), dtype=torch.float32)

        return F.adjust_lighting(image, alpha, self.eigen_value, self.eigen_vector), target


@TRANSFORMS.register()
class RandomAffine(torch.nn.Module):
    """Before BC-Break of resample.
    Random affine transformation of the image keeping center invariant.
    The image can be a PIL Image or a Tensor, in which case it is expected
    to have [..., H, W] shape, where ... means an arbitrary number of leading dimensions.
    Args:
        degrees (sequence or float or int): Range of degrees to select from.
            If degrees is a number instead of sequence like (min, max), the range of degrees
            will be (-degrees, +degrees). Set to 0 to deactivate rotations.
        translate (tuple, optional): tuple of maximum absolute pixels for horizontal
            and vertical translations. For example translate=(a, b), then horizontal shift
            is randomly sampled in the range a < dx < a and vertical shift is
            randomly sampled in the range b < dy < b. Will not translate by default.
        scale (tuple, optional): scaling factor interval, e.g (a, b), then scale is
            randomly sampled from the range a <= scale <= b. Will keep original scale by default.
        shear (sequence or float or int, optional): Range of degrees to select from.
            If shear is a number, a shear parallel to the x axis in the range (-shear, +shear)
            will be applied. Else if shear is a tuple or list of 2 values a shear parallel to the x axis in the
            range (shear[0], shear[1]) will be applied. Else if shear is a tuple or list of 4 values,
            a x-axis shear in (shear[0], shear[1]) and y-axis shear in (shear[2], shear[3]) will be applied.
            Will not apply shear by default.
        resample (int, optional): An optional resampling filter. See `filters`_ for more information.
            If omitted, or if the image has mode "1" or "P", it is set to ``PIL.Image.NEAREST``.
            If input is Tensor, only ``PIL.Image.NEAREST`` and ``PIL.Image.BILINEAR`` are supported.
        fillcolor (tuple or int): Optional fill color (Tuple for RGB Image and int for grayscale) for the area
            outside the transform in the output image (Pillow>=5.0.0). This option is not supported for Tensor
            input. Fill value for the area outside the transform in the output image is always 0.
    .. _filters: https://pillow.readthedocs.io/en/latest/handbook/concepts.html#filters
    """

    def __init__(self, degrees, translate=None, scale=None, shear=None, ignore_x=-2):
        super().__init__()
        self.ignore_x = ignore_x
        self.degrees = _setup_angle(degrees, name="degrees", req_sizes=(2,))

        if translate is not None:
            _check_sequence_input(translate, "translate", req_sizes=(2,))
        self.translate = translate

        if scale is not None:
            _check_sequence_input(scale, "scale", req_sizes=(2,))
            for s in scale:
                if s <= 0:
                    raise ValueError("scale values should be positive")
        self.scale = scale

        if shear is not None:
            self.shear = _setup_angle(shear, name="shear", req_sizes=(2, 4))
        else:
            self.shear = shear

    @staticmethod
    def get_params(
            degrees,
            translate,
            scale_ranges,
            shears
    ):
        """Get parameters for affine transformation
        Returns:
            params to be passed to the affine transformation
        """
        angle = float(torch.empty(1).uniform_(float(degrees[0]), float(degrees[1])).item())
        if translate is not None:
            max_dx = float(translate[0])
            max_dy = float(translate[1])
            tx = int(round(torch.empty(1).uniform_(-max_dx, max_dx).item()))
            ty = int(round(torch.empty(1).uniform_(-max_dy, max_dy).item()))
            translations = (tx, ty)
        else:
            translations = (0, 0)

        if scale_ranges is not None:
            scale = float(torch.empty(1).uniform_(scale_ranges[0], scale_ranges[1]).item())
        else:
            scale = 1.0

        shear_x = shear_y = 0.0
        if shears is not None:
            shear_x = float(torch.empty(1).uniform_(shears[0], shears[1]).item())
            if len(shears) == 4:
                shear_y = float(torch.empty(1).uniform_(shears[2], shears[3]).item())

        shear = (shear_x, shear_y)

        return angle, translations, scale, shear

    def forward(self, image, target):
        img_size = F._get_image_size(image)
        ret = self.get_params(self.degrees, self.translate, self.scale, self.shear)
        image = F.affine(image, *ret, resample=Image.LINEAR, fillcolor=0)

        if target is None or isinstance(target, str):
            return image, target
        elif isinstance(target, dict):  # To keep BC
            if 'keypoints' in target.keys():
                target['keypoints'] = F_kp.affine(target['keypoints'], *ret,
                                                  height=img_size[1], width=img_size[0], ignore_x=self.ignore_x)
            # if 'padding_mask' in target.keys():
            #     pass
            if 'segmentation_mask' in target.keys():
                target['segmentation_mask'] = F.affine(target['segmentation_mask'], *ret,
                                                       resample=Image.NEAREST, fillcolor=255)
        else:
            target = F.affine(target, *ret, resample=Image.NEAREST, fillcolor=255)

        return image, target


@TRANSFORMS.register()
class ToXOffset(torch.nn.Module):
    def __init__(self, num_points, image_size, max_lanes, ignore_x=-2):
        super().__init__()
        assert isinstance(image_size, tuple or list)
        self.img_h, self.img_w = image_size
        self.num_points = num_points
        self.num_offsets = num_points
        self.num_strips = num_points - 1
        self.strip_size = self.img_h / self.num_strips
        self.max_lanes = max_lanes
        self.ignore_x = ignore_x
        self.offsets_ys = np.arange(self.img_h, -1, -self.strip_size)

    def clip_out_of_image(self, x):
        # TODO: len(x) is 0
        num_lanes = x.shape[0]
        lanes_ = []
        for i in range(num_lanes):
            lane = x[i]
            lane_temp = [[point[0], point[1]] for point in lane if point[0] != self.ignore_x]
            lanes_.append(lane_temp)
        # for isse in lanes_:
        #     print(isse)
        # quit(0)
        return lanes_

    def filter_lane(self, lane):
        assert lane[-1][1] <= lane[0][1]
        filtered_lane = []
        used = set()
        for p in lane:
            if p[1] not in used:
                filtered_lane.append(p)
                used.add(p[1])

        return filtered_lane

    def transform_annotation(self, lanes, img_wh=None):
        img_w, img_h = self.img_w, self.img_h

        old_lanes = lanes

        # removing lanes with less than 2 points
        old_lanes = filter(lambda x: len(x) > 1, old_lanes)
        # sort lane points by Y (bottom to top of the image)
        old_lanes = [sorted(lane, key=lambda x: -x[1]) for lane in old_lanes]
        # remove points with same Y (keep first occurrence)
        old_lanes = [self.filter_lane(lane) for lane in old_lanes]
        # normalize the annotation coordinates
        old_lanes = [[[x * self.img_w / float(img_w), y * self.img_h / float(img_h)] for x, y in lane]
                     for lane in old_lanes]
        # create tranformed annotations
        lanes = np.ones((self.max_lanes, 2 + 1 + 1 + 1 + self.num_offsets),
                        dtype=np.float32) * -1e5  # 2 scores, 1 start_y, 1 start_x, 1 length, S+1 coordinates
        # lanes are invalid by default
        lanes[:, 0] = 1
        lanes[:, 1] = 0
        for lane_idx, lane in enumerate(old_lanes):
            try:
                xs_outside_image, xs_inside_image = self.sample_lane(lane, self.offsets_ys)
            except AssertionError:
                continue
            if len(xs_inside_image) == 0:
                continue
            all_xs = np.hstack((xs_outside_image, xs_inside_image))
            lanes[lane_idx, 0] = 0
            lanes[lane_idx, 1] = 1
            lanes[lane_idx, 2] = len(xs_outside_image) / self.num_strips
            lanes[lane_idx, 3] = xs_inside_image[0]
            lanes[lane_idx, 4] = len(xs_inside_image)
            lanes[lane_idx, 5:5 + len(all_xs)] = all_xs

        # new_anno = {'label': lanes, 'old_anno': anno}
        return lanes

    def sample_lane(self, points, sample_ys):
        # this function expects the points to be sorted
        points = np.array(points)
        if not np.all(points[1:, 1] < points[:-1, 1]):
            raise Exception('Annotaion points have to be sorted')
        x, y = points[:, 0], points[:, 1]

        # interpolate points inside domain
        assert len(points) > 1
        interp = InterpolatedUnivariateSpline(y[::-1], x[::-1], k=min(3, len(points) - 1))
        domain_min_y = y.min()
        domain_max_y = y.max()
        sample_ys_inside_domain = sample_ys[(sample_ys >= domain_min_y) & (sample_ys <= domain_max_y)]
        assert len(sample_ys_inside_domain) > 0
        interp_xs = interp(sample_ys_inside_domain)

        # extrapolate lane to the bottom of the image with a straight line using the 2 points closest to the bottom
        two_closest_points = points[:2]
        extrap = np.polyfit(two_closest_points[:, 1], two_closest_points[:, 0], deg=1)
        extrap_ys = sample_ys[sample_ys > domain_max_y]
        extrap_xs = np.polyval(extrap, extrap_ys)
        all_xs = np.hstack((extrap_xs, interp_xs))

        # separate between inside and outside points
        inside_mask = (all_xs >= 0) & (all_xs < self.img_w)
        xs_inside_image = all_xs[inside_mask]
        xs_outside_image = all_xs[~inside_mask]

        return xs_outside_image, xs_inside_image

    def forward(self, image, target):
        keypoints = target['keypoints']
        cilp_lanes = self.clip_out_of_image(keypoints)
        offsets = self.transform_annotation(cilp_lanes)
        #image = np.asarray(image) / 255.

        return image, offsets
