import os

from .image_folder_lane_base import ImageFolderLaneBase
from .builder import DATASETS


# Visualization version of CULane
@DATASETS.register()
class CULaneVis(ImageFolderLaneBase):
    def __init__(self, root_dataset, root_output, root_keypoint, image_set, transforms=None,
                 keypoint_process_fn=None, use_gt=True):
        super().__init__(root_dataset, root_output, transforms, keypoint_process_fn)
        self.image_set = image_set

        self._check()

        # Data list
        with open(os.path.join(root_dataset, 'lists', image_set + '.txt'), "r") as f:
            contents = [x.strip() for x in f.readlines()]

        # Load filenames
        if image_set == 'test' or image_set == 'val':  # Test
            self.images = [os.path.join(root_dataset, x + '.jpg') for x in contents]
            if use_gt:
                self.gt_keypoints = [os.path.join(root_dataset, x + '.lines.txt') for x in contents]
            self.filenames = [x + '.jpg' for x in contents]
            if root_keypoint is not None:
                self.keypoints = [os.path.join(root_keypoint, x + '.lines.txt') for x in contents]
        else:  # Train
            self.images = [os.path.join(root_dataset, x[:x.find(' ')] + '.jpg') for x in contents]
            if use_gt:
                self.gt_keypoints = [os.path.join(root_dataset, x[:x.find(' ')] + '.lines.txt') for x in contents]
            self.filenames = [x[:x.find(' ')] + '.jpg' for x in contents]
            if root_keypoint is not None:
                self.keypoints = [os.path.join(root_keypoint, x + '.lines.txt') for x in contents]

        self.make_sub_dirs()

        assert len(self.images) == len(self.gt_keypoints)
        if self.keypoints is not None:
            assert len(self.images) == len(self.keypoints)

    def _check(self):
        # Checks
        if self.image_set not in ['train', 'val', 'test']:
            raise ValueError
        assert self.output_dir != self.root, 'Avoid overwriting your dataset!'
