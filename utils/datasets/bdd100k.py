import torchvision
import os
import numpy as np
from PIL import Image


# BDD100K direct loading (work with the segmentation style lists)
class CULane(torchvision.datasets.VisionDataset):
    def __init__(self, root, image_set, transforms=None, transform=None, target_transform=None,
                 ppl=0, gap=0, start=0):
        super().__init__(root, transforms, transform, target_transform)
        pass

    def __getitem__(self, index):
        # Return x (input image) & y (L lane with N coordinates (x, y) as np.array (L x N x 2))
        # Empty coordinates are marked by (-2, -2)
        # If just testing,
        # y is the filename to store prediction
        pass

    def __len__(self):
        pass
