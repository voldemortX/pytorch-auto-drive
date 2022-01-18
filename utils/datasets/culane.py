import os
import pickle
import numpy as np
from tqdm import tqdm

from .utils import LaneKeypointDataset
from .builder import DATASETS


# CULane direct loading (work with the segmentation style lists)
@DATASETS.register()
class CULane(LaneKeypointDataset):
    def __init__(self, root, image_set, transforms=None, transform=None, target_transform=None,
                 ppl=31, gap=10, start=290, padding_mask=False, is_process=True):
        super().__init__(root, transforms, transform, target_transform, ppl, gap, start, padding_mask, image_set,
                         is_process)

        self._check()

        # Data list
        with open(os.path.join(root, 'lists', image_set + '.txt'), "r") as f:
            contents = [x.strip() for x in f.readlines()]

        # Load filenames
        if image_set == 'test' or image_set == 'val':  # Test
            self.images = [os.path.join(root, x + '.jpg') for x in contents]
            self.targets = [os.path.join('./output', x + '.lines.txt') for x in contents]
        else:  # Train
            self.images = [os.path.join(root, x[:x.find(' ')] + '.jpg') for x in contents]
            self.targets = []
            print('Loading targets into memory...')
            processed_file = os.path.join(root, 'train_processed_targets')
            if os.path.exists(processed_file):
                with open(processed_file, 'rb') as f:
                    self.targets = pickle.load(f)
            else:
                print('Pre-processing will only be performed for 1 time, please wait ~10 minutes.')
                for x in tqdm(contents):
                    with open(os.path.join(root, x[:x.find(' ')] + '.lines.txt'), 'r') as f:
                        self.targets.append(self._load_target(f.readlines()))
                with open(processed_file, 'wb') as f:
                    pickle.dump(self.targets, f)
            print('Loading complete.')

        assert len(self.targets) == len(self.images)

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
