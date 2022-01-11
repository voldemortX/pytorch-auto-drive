import os
import torch
from tqdm import tqdm
from abc import abstractmethod
from PIL import Image
if torch.__version__ >= '1.6.0':
    from torch.cuda.amp import autocast
else:
    from ..torch_amp_dummy import autocast

from .base import BaseVisualizer, BaseVideoVisualizer, get_collate_fn
from ..datasets import DATASETS
from ..transforms import TRANSFORMS, ToTensor, functional as F
from ..vis_utils import segmentation_visualize_batched, save_images, \
    find_transform_by_name, get_transform_attr_by_name, tensor_image_to_numpy


def seg_label_process_fn(label):
    # Open and process a common seg label from filename
    label = Image.open(label)
    label = ToTensor.label_to_tensor(label)

    return label


class SegVisualizer(BaseVisualizer):
    dataset_tensor_statistics = ['colors']

    @torch.no_grad()
    def seg_inference(self, images, original_size=None, pad_crop=False):
        # Segmentation methods have simple and unified output formats,
        # same simple post-process will suffice
        if original_size is None:
            original_size = self._cfg['original_size']
        with autocast(self._cfg['mixed_precision']):
            results = self.model(images)['out']
            if pad_crop:  # VOC style transform
                labels = torch.nn.functional.interpolate(results, size=images.shape[2:],
                                                         mode='bilinear', align_corners=True)
                labels = F.crop(labels, 0, 0, original_size[0], original_size[1])
            else:
                labels = torch.nn.functional.interpolate(results, size=original_size,
                                                         mode='bilinear', align_corners=True)
        return labels.argmax(1)

    @abstractmethod
    def run(self, *args, **kwargs):
        pass

    @abstractmethod
    def get_loader(self, *args, **kwargs):
        pass


class SegDir(SegVisualizer):
    def __init__(self, cfg):
        super().__init__(cfg)
        os.makedirs(self._cfg['save_path'], exist_ok=True)
        self.pad_crop = find_transform_by_name(cfg['test_augmentation'], 'ZeroPad')
        self.id_map = None
        if self._cfg['map_id']:
            id_map = get_transform_attr_by_name(cfg['test_augmentation'], 'LabelMap', attr='label_id_map')
            self.id_map = torch.tensor(id_map)

    def get_loader(self, cfg):
        if 'vis_dataset' in cfg.keys():
            dataset_cfg = cfg['vis_dataset']
        else:
            dataset_cfg = dict(
                name='ImageFolderDataset',
                root_image=self._cfg['image_path'],
                root_target=self._cfg['target_path'],
                root_output=self._cfg['save_path'],
                image_suffix=self._cfg['image_suffix'],
                target_suffix=self._cfg['target_suffix']
            )
        dataset = DATASETS.from_dict(dataset_cfg,
                                     transforms=TRANSFORMS.from_dict(cfg['test_augmentation']),
                                     target_process_fn=seg_label_process_fn)

        collate_fn = get_collate_fn('dict_collate_fn')  # Use dicts for customized target
        dataloader = torch.utils.data.DataLoader(dataset=dataset,
                                                 batch_size=self._cfg['batch_size'],
                                                 collate_fn=collate_fn,
                                                 num_workers=self._cfg['workers'],
                                                 shuffle=False)

        return dataloader, cfg['dataset']['name']

    def run(self):
        for imgs, original_imgs, targets in tqdm(self.dataloader):
            filenames = [i['filename'] for i in targets]
            targets = [i['target'] for i in targets]
            if targets.count(None) == len(targets):
                targets = None
            else:
                targets = torch.stack(targets)
            if self._cfg['pred']:  # Inference
                imgs = imgs.to(self.device)
                original_imgs = original_imgs.to(self.device)
                targets = self.seg_inference(imgs, original_imgs.shape[2:], pad_crop=self.pad_crop)
            elif self.id_map is not None:
                targets[targets >= self.id_map.shape[0]] = 0
                targets = self.id_map[targets]
            results = segmentation_visualize_batched(original_imgs,
                                                     targets,
                                                     colors=self._cfg['colors'],
                                                     std=None, mean=None)
            save_images(results, filenames=filenames)


class SegVideo(BaseVideoVisualizer, SegVisualizer):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.pad_crop = find_transform_by_name(cfg['test_augmentation'], 'ZeroPad')

    def run(self):
        # Must do inference
        for imgs, original_imgs in tqdm(self.dataloader):
            targets = None
            if self._cfg['pred']:  # Inference
                imgs = imgs.to(self.device)
                original_imgs = original_imgs.to(self.device)
                targets = self.seg_inference(imgs, original_imgs.shape[2:], pad_crop=self.pad_crop)
            results = segmentation_visualize_batched(original_imgs,
                                                     targets,
                                                     colors=self._cfg['colors'],
                                                     std=None, mean=None)
            np_results = tensor_image_to_numpy(results)[..., [2, 1, 0]]
            for j in range(np_results.shape[0]):
                self.writer.write(np_results[j])


class SegDataset(SegVisualizer):
    def __init__(self, cfg):
        super().__init__(cfg)

    def get_loader(self, cfg):
        pass

    def run(self):
        pass
