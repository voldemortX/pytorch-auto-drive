import torchvision
import os
import numpy as np
from PIL import Image


# CULane direct loading (work with the segmentation style lists)
class CULane(torchvision.datasets.VisionDataset):
    def __init__(self, root, image_set, transforms=None, transform=None, target_transform=None,
                 ppl=31, gap=10, start=290):
        super().__init__(root, transforms, transform, target_transform)
        self.ppl = ppl
        self.gap = gap
        self.start = start  # y coordinate to start annotation
        self.test = False

        # Checks
        if not os.path.exists('./output'):
            os.makedirs('./output')
        if image_set not in ['train', 'val', 'test']:
            raise ValueError

        # Data list
        with open(os.path.join(root, 'lists', image_set + '.txt'), "r") as f:
            contents = [x.strip() for x in f.readlines()]

        # Load filenames
        if image_set == 'test' or image_set == 'val':  # Test
            self.test = True
            self.images = [os.path.join(root, x + '.jpg') for x in contents]
            self.targets = [os.path.join('./output', x + '.lines.txt') for x in contents]
        else:  # Train
            self.images = [os.path.join(root, x[:x.find(' ')] + '.jpg') for x in contents]
            self.targets = [os.path.join(root, x[:x.find(' ')] + '.lines.txt') for x in contents]

        assert len(self.targets) == len(self.images)

    def __getitem__(self, index):
        # Return x (input image) & y (L lane with N coordinates (x, y) as np.array (L x N x 2))
        # Empty coordinates are marked by (-2, y)
        # If just testing,
        # y is the filename to store prediction
        img = Image.open(self.images[index]).convert('RGB')
        if self.test is True:
            target = self.targets[index]
        else:
            with open(self.targets[index], 'r') as f:
                target = self._load_target(f.readlines())

        # Transforms
        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.images)

    def _load_target(self, lines):
        # Read file content to lists (file content could be empty or variable number of lanes)
        target = np.array([[[-2.0, self.start + i * self.gap] for i in range(self.ppl)]
                          for _ in range(len(lines))], dtype=np.float32)
        for i in range(len(lines)):  # lines=[] will end this immediately
            temp = [float(k) for k in lines[i].strip().split(' ')]
            for j in range(int(len(temp) / 2)):
                x = temp[2 * j]
                y = temp[2 * j + 1]
                target[i][target[i][:, 1] == y] = [x, y]

        return target

    @staticmethod
    def load_target_xy(lines):
        # A direct loading of JSON file to a list of N x 2 numpy arrays
        target = []
        for line in lines:
            temp = [float(x) for x in line.strip().split(' ')]
            target.append(np.array(temp).reshape(-1, 2))

        return target
