import torch
from mmcv import VideoReader

from ..transforms import Compose
from .builder import DATASETS


# Load a video for inference
@DATASETS.register()
class VideoLoader(object):
    def __init__(self, filename, transforms=None, batch_size=1, *args, **kwargs):
        # Don't need ToTensor here
        self.transforms = Compose(transforms=[t for t in transforms.transforms if t.__class__.__name__ != 'ToTensor'])
        self.batch_size = batch_size
        self.video = VideoReader(filename)
        self.resolution = self.video.resolution
        self.fps = self.video.fps
        self.i = 0

    def __next__(self):
        # Return transformed images / original images
        # Numpy can suffer a index OOB
        if self.i >= len(self):
            raise StopIteration

        images_numpy = self.video[self.i * self.batch_size: (self.i + 1) * self.batch_size]
        images = torch.stack([torch.from_numpy(img) for img in images_numpy])
        images = images.permute(0, 3, 1, 2) / 255.0  # BHWC-rgb uint8 -> BCHW-rgb float
        original_images = images.clone()

        # Transforms
        if self.transforms is not None:
            images = self.transforms(images)

        self.i += 1

        return images, original_images

    def __iter__(self):
        return self

    def __len__(self):
        return len(self.video) // self.batch_size
