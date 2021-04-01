import torch
import torchvision
from PIL import Image
from transforms import functional_pil as f_pil


def generate_lane_label_dict(target):
    # target: {'keypoints': Tensor, L x N x 2, ...}
    # Although non-existent keypoints are marked as (-2, y), it is safer to check with > 0

    # Drop invalid lanes (lanes with less than 2 keypoints are seen as invalid)
    valid_lanes = (target['keypoints'][:, :, 0] > 0).sum(dim=-1) >= 2
    target['keypoints'] = target['keypoints'][valid_lanes]

    # Append lowest & highest y coordinates, labels (all 1)
    target['lowers'] = [l[l[:, 0] > 0][:, 1].min() for l in target['keypoints']]  # Looks better than giving MIN values
    target['uppers'] = [l[l[:, 0] > 0][:, 1].max() for l in target['keypoints']]
    target['labels'] = torch.ones(target['keypoints'].shape[0], device=target['keypoints'].device, dtype=torch.int64)
    
    return target


# Lanes as keypoints
class LaneKeypointDataset(torchvision.datasets.VisionDataset):
    def __init__(self, root, transforms, transform, target_transform,
                 ppl, gap, start, padding_mask, process_points):
        super().__init__(root, transforms, transform, target_transform)
        self.ppl = ppl  # Sampled points-per-lane
        self.gap = gap  # y gap between sample points
        self.start = start  # y coordinate to start annotation
        self.padding_mask = padding_mask  # Padding mask for transformer
        self.process_points = process_points  # Add lowest & highest y coordinates, lane class labels
        self.images = []  # placeholder
        self.targets = []  # placeholder

    def __getitem__(self, index):
        # Return x (input image) & y (L lane with N coordinates (x, y) as np.array (L x N x 2))
        # Invalid coordinates are marked by (-2, y)
        # If just testing,
        # y is the filename to store prediction
        img = Image.open(self.images[index]).convert('RGB')
        target = self.targets[index]

        # Load as dict
        if self.padding_mask or self.process_points:
            if type(target) == str:
                print('Testing does not require target padding_mask or process_point!')
                raise ValueError
            else:
                target = {'keypoints': target}

        # Add padding mask
        if self.padding_mask:
            target['padding_mask'] = Image.new("RGB", f_pil._get_image_size(img), (0, 0, 0))

        # Transforms
        if self.transforms is not None:
            img, target = self.transforms(img, target)

        # Post-process
        target = generate_lane_label_dict(target)

        return img, target

    def __len__(self):
        return len(self.images)
