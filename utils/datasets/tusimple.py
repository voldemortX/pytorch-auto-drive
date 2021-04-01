import os
import ujson as json
import numpy as np
from tqdm import tqdm
from .utils import LaneKeypointDataset


# TuSimple direct loading
class TuSimple(LaneKeypointDataset):
    def __init__(self, root, image_set, transforms=None, transform=None, target_transform=None,
                 ppl=56, gap=10, start=160, padding_mask=False, process_points=False):
        super().__init__(root, transforms, transform, target_transform, ppl, gap, start, padding_mask, process_points)

        # Checks
        if not os.path.exists('./output'):
            os.makedirs('./output')
        if image_set not in ['train', 'val', 'test']:
            raise ValueError

        # Data list
        with open(os.path.join(root, 'lists', image_set + '.txt'), "r") as f:
            contents = [x.strip() for x in f.readlines()]

        # Load image filenames and lanes
        if image_set == 'test' or image_set == 'val':  # Test
            self.images = [os.path.join(root, 'clips', x + '.jpg') for x in contents]
            self.targets = [os.path.join(root, 'clips', x + '.jpg') for x in contents]
        else:  # Train
            self.images = [os.path.join(root, 'clips', x[:x.find(' ')] + '.jpg') for x in contents]

            # Load target lanes (small dataset, directly load all of them in the memory)
            print('Loading targets into memory...')
            target_files = [os.path.join(root, 'label_data_0313.json'),
                            os.path.join(root, 'label_data_0601.json')]
            json_contents = self.concat_jsons(target_files)
            self.targets = []
            for i in tqdm(range(len(json_contents))):
                lines = json_contents[i]['lanes']
                h_samples = json_contents[i]['h_samples']
                temp = np.array([[[-2.0, self.start + j * self.gap] for j in range(self.ppl)]
                                for _ in range(len(lines))], dtype=np.float32)
                for j in range(len(h_samples)):
                    for k in range(len(lines)):
                        temp[k][temp[k][:, 1] == h_samples[j]] = [float(lines[k][j]), h_samples[j]]
                self.targets.append(temp)

        assert len(self.targets) == len(self.images)

    @staticmethod
    def concat_jsons(filenames):
        # Concat tusimple lists in jsons (actually only each line is json)
        results = []
        for filename in filenames:
            with open(filename, 'r') as f:
                results += [json.loads(x.strip()) for x in f.readlines()]

        return results
