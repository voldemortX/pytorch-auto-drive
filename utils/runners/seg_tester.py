import os
import torch
from tqdm import tqdm
if torch.__version__ >= '1.6.0':
    from torch.cuda.amp import autocast
else:
    from ..torch_amp_dummy import autocast

from ..seg_utils import ConfusionMatrix
from .base import BaseTester


class SegTester(BaseTester):
    def __init__(self, cfg):
        super().__init__(cfg, map_dataset_statics=['categories'])

    def run(self):
        acc, iou, res_str = self.test_one_set(self.dataloader, self.device, self.model,
                                              self._cfg['num_classes'], self._cfg['categories'],
                                              self._cfg['original_size'], self._cfg['encoder_size'],
                                              self._cfg['mixed_precision'],
                                              self._cfg['selector'], self._cfg['eval_classes'],
                                              self._cfg['encoder_only'])
        self.write_mp_log('log.txt', self._cfg['exp_name'] + ': ' + str(iou) + '\n')
        prefix = 'val' if self._cfg['state'] == 1 else 'custom_state_' + str(self._cfg['state'])
        with open(os.path.join(self._cfg['exp_dir'], prefix + '_result.txt'), 'w') as f:
            f.write(res_str)

    @staticmethod
    @torch.no_grad()
    def test_one_set(loader, device, net, num_classes, categories, output_size, labels_size, mixed_precision,
                     selector=None, classes=None, encoder_only=False):
        # Copied and modified from torch/vision/references/segmentation
        # Evaluate on 1 data_loader
        # Use selector & classes to select part of the classes as metric (for SYNTHIA)
        net.eval()
        conf_mat = ConfusionMatrix(num_classes)
        for image, target in tqdm(loader):
            image, target = image.to(device), target.to(device)
            with autocast(mixed_precision):
                output = net(image)['out']
                if encoder_only:
                    target = target.unsqueeze(0)
                    if target.dtype not in (torch.float32, torch.float64):
                        target = target.to(torch.float32)
                    target = torch.nn.functional.interpolate(target, size=labels_size, mode='nearest')
                    target = target.to(torch.int64)
                    target = target.squeeze(0)
                else:
                    output = torch.nn.functional.interpolate(output, size=output_size, mode='bilinear',
                                                             align_corners=True)
                conf_mat.update(target.flatten(), output.argmax(1).flatten())
        conf_mat.reduce_from_all_processes()

        acc_global, acc, iu = conf_mat.compute()
        res_str = (
            'All classes: {}\n'
            'Pixel acc: {:.2f}\n'
            'Pixel acc (per-class): {}\n'
            'IoU (per-class): {}\n'
            'Mean IoU: {:.2f}\n'
            'Mean IoU-{}: {:.2f}').format(
            categories,
            acc_global.item() * 100,
            ['{:.2f}'.format(i) for i in (acc * 100).tolist()],
            ['{:.2f}'.format(i) for i in (iu * 100).tolist()],
            iu.mean().item() * 100,
            -1 if classes is None else classes,
            -1 if selector is None else iu[selector].mean().item() * 100)
        print(res_str)

        if selector is None:
            iou = iu.mean().item() * 100
        else:
            iou = iu[selector].mean().item() * 100

        return acc_global.item() * 100, iou, res_str
