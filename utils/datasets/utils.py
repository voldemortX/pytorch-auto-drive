import torch
import torchvision
from PIL import Image
from transforms import functional_pil as f_pil
from torch._six import container_abcs, string_classes, int_classes
from torch.utils.data._utils.collate import default_collate_err_msg_format, np_str_obj_array_pattern


def dict_collate_fn(batch):
    # To keep each image's label as separate dictionaries, default pytorch behaviour will stack each key
    # Only modified one line of the pytorch 1.6.0 default collate function

    elem = batch[0]
    elem_type = type(elem)
    if isinstance(elem, torch.Tensor):
        out = None
        if torch.utils.data.get_worker_info() is not None:
            # If we're in a background process, concatenate directly into a
            # shared memory tensor to avoid an extra copy
            numel = sum([x.numel() for x in batch])
            storage = elem.storage()._new_shared(numel)
            out = elem.new(storage)
        return torch.stack(batch, 0, out=out)
    elif elem_type.__module__ == 'numpy' and elem_type.__name__ != 'str_' \
            and elem_type.__name__ != 'string_':
        elem = batch[0]
        if elem_type.__name__ == 'ndarray':
            # array of string classes and object
            if np_str_obj_array_pattern.search(elem.dtype.str) is not None:
                raise TypeError(default_collate_err_msg_format.format(elem.dtype))

            return dict_collate_fn([torch.as_tensor(b) for b in batch])
        elif elem.shape == ():  # scalars
            return torch.as_tensor(batch)
    elif isinstance(elem, float):
        return torch.tensor(batch, dtype=torch.float64)
    elif isinstance(elem, int_classes):
        return torch.tensor(batch)
    elif isinstance(elem, string_classes):
        return batch
    elif isinstance(elem, container_abcs.Mapping):
        return batch  # !Only modified this line
    elif isinstance(elem, tuple) and hasattr(elem, '_fields'):  # namedtuple
        return elem_type(*(dict_collate_fn(samples) for samples in zip(*batch)))
    elif isinstance(elem, container_abcs.Sequence):
        # check to make sure that the elements in batch have consistent size
        it = iter(batch)
        elem_size = len(next(it))
        if not all(len(elem) == elem_size for elem in it):
            raise RuntimeError('each element in list of batch should be of equal size')
        transposed = zip(*batch)
        return [dict_collate_fn(samples) for samples in transposed]

    raise TypeError(default_collate_err_msg_format.format(elem_type))


def generate_lane_label_dict(target):
    # target: {'keypoints': Tensor, L x N x 2, ...}
    # Although non-existent keypoints are marked as (-2, y), it is safer to check with > 0

    # Drop invalid lanes (lanes with less than 2 keypoints are seen as invalid)
    valid_lanes = (target['keypoints'][:, :, 0] > 0).sum(dim=-1) >= 2
    target['keypoints'] = target['keypoints'][valid_lanes]

    # Append lowest & highest y coordinates (coordinates start at top-left corner), labels (all 1)
    # Looks better than giving MIN values
    target['lowers'] = torch.stack([l[l[:, 0] > 0][:, 1].max() for l in target['keypoints']])
    target['uppers'] = torch.stack([l[l[:, 0] > 0][:, 1].min() for l in target['keypoints']])
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
            target['padding_mask'] = Image.new("L", f_pil._get_image_size(img), 0)

        # Transforms
        if self.transforms is not None:
            img, target = self.transforms(img, target)

        # Post-process
        if self.process_points:
            target = generate_lane_label_dict(target)

        return img, target

    def __len__(self):
        return len(self.images)
