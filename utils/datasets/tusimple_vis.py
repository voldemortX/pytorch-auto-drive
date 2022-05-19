import os
import numpy as np
from tqdm import tqdm

from .image_folder_lane_base import ImageFolderLaneBase
from .tusimple import TuSimple
from .builder import DATASETS


def dummy_keypoint_process_fn(label):
    return label


# Visualization version of CULane
@DATASETS.register()
class TuSimpleVis(ImageFolderLaneBase):
    def __init__(self, root_dataset, root_output, keypoint_json, image_set, transforms=None,
                 keypoint_process_fn=None, use_gt=True):
        super().__init__(root_dataset, root_output, transforms, dummy_keypoint_process_fn)
        self.image_set = image_set

        self._check()

        # Data list
        with open(os.path.join(root_dataset, 'lists', image_set + '.txt'), "r") as f:
            contents = [x.strip() for x in f.readlines()]

        # Load filenames
        if image_set == 'test' or image_set == 'val':  # Test
            self.images = [os.path.join(root_dataset, 'clips', x + '.jpg') for x in contents]
            self.filenames = [os.path.join('clips', x + '.jpg') for x in contents]
            if use_gt:
                self.gt_keypoints = []
                target_files = [os.path.join(root_dataset, 'label_data_0531.json')]
                if image_set == 'test':
                    target_files = [os.path.join(root_dataset, 'test_label.json')]
                json_contents = TuSimple.concat_jsons(target_files)
                self.gt_keypoints = self.preload_tusimple_labels(json_contents)
            if keypoint_json is not None:
                json_contents = TuSimple.concat_jsons([keypoint_json])
                self.keypoints = self.preload_tusimple_labels(json_contents)
        else:  # Train
            self.images = [os.path.join(root_dataset, 'clips', x[:x.find(' ')] + '.jpg') for x in contents]
            self.filenames = [os.path.join('clips', x[:x.find(' ')] + '.jpg') for x in contents]
            if use_gt:
                self.gt_keypoints = []
                target_files = [os.path.join(root_dataset, 'label_data_0313.json'),
                                os.path.join(root_dataset, 'label_data_0601.json')]
                json_contents = TuSimple.concat_jsons(target_files)
                self.gt_keypoints = self.preload_tusimple_labels(json_contents)
            if keypoint_json is not None:
                json_contents = TuSimple.concat_jsons([keypoint_json])
                self.keypoints = self.preload_tusimple_labels(json_contents)

        self.make_sub_dirs()

        assert len(self.images) == len(self.gt_keypoints)
        if self.keypoints is not None:
            assert len(self.images) == len(self.keypoints)

    def _check(self):
        # Checks
        if self.image_set not in ['train', 'val', 'test']:
            raise ValueError
        assert self.output_dir != self.root, 'Avoid overwriting your dataset!'

    @staticmethod
    def preload_tusimple_labels(json_contents):
        # Load a TuSimple label json's content
        print('Loading json annotation/prediction...')
        targets = []
        for i in tqdm(range(len(json_contents))):
            lines = json_contents[i]['lanes']
            h_samples = json_contents[i]['h_samples']
            temp = []
            for j in range(len(lines)):
                temp.append(np.array([[float(x), float(y)] for x, y in zip(lines[j], h_samples)]))
            targets.append(temp)

        return targets
