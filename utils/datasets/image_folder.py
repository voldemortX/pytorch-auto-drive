import torchvision
import os
from PIL import Image

from ..transforms import functional as F, ToTensor
from .builder import DATASETS
from .image_folder_lane_base import ImageFolderLaneBase


# Load a directory of images for inference
@DATASETS.register()
class ImageFolderDataset(torchvision.datasets.VisionDataset):
    def __init__(self, root_image, root_output, root_target=None, transforms=None,
                 target_process_fn=None, image_suffix='', target_suffix=''):
        super().__init__(root_image, transforms, None, None)
        self.output_dir = root_output
        self.filenames = []
        self.images = []
        self.targets = None if root_target is None else []
        self.target_process_fn = target_process_fn
        for filename in sorted(os.listdir(root_image)):
            suffix_pos = filename.rfind(image_suffix)
            if suffix_pos != -1:
                middle_name = filename[:suffix_pos]
                self.filenames.append(filename)
                self.images.append(os.path.join(root_image, filename))
                if self.targets is not None:
                    self.targets.append(os.path.join(root_target, middle_name + target_suffix))

    def __getitem__(self, index):
        # Return transformed image / original image / save filename / label (if exist)
        img = Image.open(self.images[index]).convert('RGB')
        filename = os.path.join(self.output_dir, self.filenames[index])
        original_img = F.to_tensor(img).clone()

        # Transforms
        if self.transforms is not None:
            img = self.transforms(img)

        # Process potential target
        target = None
        if self.targets is not None:
            target = self.target_process_fn(self.targets[index])

        return img, original_img, {
            'filename': filename,
            'target': target
        }

    def __len__(self):
        return len(self.images)


# Load a directory of images for lane inference
@DATASETS.register()
class ImageFolderLaneDataset(ImageFolderLaneBase):
    def __init__(self, root_image, root_output, root_keypoint=None, root_gt_keypoint=None, root_mask=None,
                 transforms=None, keypoint_process_fn=None,
                 image_suffix='', keypoint_suffix='.txt', gt_keypoint_suffix='.txt', mask_suffix=''):
        super().__init__(root_image, root_output, transforms, keypoint_process_fn)
        self.keypoints = None if root_keypoint is None else []
        self.gt_keypoints = None if root_gt_keypoint is None else []
        self.masks = None if root_mask is None else []
        for filename in sorted(os.listdir(root_image)):
            suffix_pos = filename.rfind(image_suffix)
            if suffix_pos != -1:
                middle_name = filename[:suffix_pos]
                self.filenames.append(filename)
                self.images.append(os.path.join(root_image, filename))
                if self.keypoints is not None:
                    self.keypoints.append(os.path.join(root_keypoint, middle_name + keypoint_suffix))
                if self.gt_keypoints is not None:
                    self.gt_keypoints.append(os.path.join(root_gt_keypoint, middle_name + gt_keypoint_suffix))
                if self.masks is not None:
                    self.masks.append(os.path.join(root_mask, middle_name + mask_suffix))
