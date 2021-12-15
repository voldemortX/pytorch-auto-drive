import os
import torch
import time
if torch.__version__ >= '1.6.0':
    from torch.cuda.amp import autocast, GradScaler
else:
    from ..torch_amp_dummy import autocast, GradScaler

from ..common import save_checkpoint
from ..seg_utils import ConfusionMatrix
from ..ddp_utils import is_main_process, is_dist_avail_and_initialized, get_world_size
from .base import BaseTrainer, DATASETS, TRANSFORMS
from .seg_tester import SegTester


class SegTrainer(BaseTrainer):
    def __init__(self, cfg, args):
        super().__init__(cfg, args, map_dataset_statics=['categories'])

    def run(self):
        # Validate and find the best snapshot
        best_mIoU = 0
        self.model.train()
        epoch = 0
        running_loss = 0.0
        loss_num_steps = int(len(self.dataloader) / 10)
        if self._cfg['mixed_precision']:
            scaler = GradScaler()

        # Training
        while epoch < self._cfg['num_epochs']:
            self.model.train()
            if self._cfg['distributed']:
                self.train_sampler.set_epoch(epoch)
            conf_mat = ConfusionMatrix(self._cfg['num_classes'])
            time_now = time.time()
            for i, data in enumerate(self.dataloader, 0):
                inputs, labels = data
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()

                with autocast(self._cfg['mixed_precision']):
                    outputs = self.model(inputs)['out']

                    if self._cfg['encoder_only']:
                        labels = labels.unsqueeze(0)
                        if labels.dtype not in (torch.float32, torch.float64):
                            labels = labels.to(torch.float32)
                        labels = torch.nn.functional.interpolate(labels, size=self._cfg['encoder_size'], mode='nearest')
                        labels = labels.to(torch.int64)
                        labels = labels.squeeze(0)
                    else:
                        outputs = torch.nn.functional.interpolate(outputs, size=self._cfg['input_size'],
                                                                  mode='bilinear', align_corners=True)
                    conf_mat.update(labels.flatten(), outputs.argmax(1).flatten())
                    loss = self.criterion(outputs, labels)

                if self._cfg['mixed_precision']:
                    scaler.scale(loss).backward()
                    scaler.step(self.optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    self.optimizer.step()

                self.lr_scheduler.step()
                running_loss += loss.item()
                running_loss = torch.tensor([running_loss], dtype=loss.dtype, device=loss.device)
                if is_dist_avail_and_initialized():
                    torch.distributed.all_reduce(running_loss)
                running_loss = (running_loss / get_world_size()).item()
                current_step_num = int(epoch * len(self.dataloader) + i + 1)

                if current_step_num % loss_num_steps == (loss_num_steps - 1):
                    print('[%d, %d] loss: %.4f' % (epoch + 1, i + 1, running_loss / loss_num_steps))
                    if is_main_process():
                        self.writer.add_scalar('training loss',
                                               running_loss / loss_num_steps,
                                               current_step_num)
                    running_loss = 0.0

                # Validate and find the best snapshot
                if current_step_num % self._cfg['val_num_steps'] == (self._cfg['val_num_steps'] - 1):
                    test_pixel_accuracy, test_mIoU = SegTester.test_one_set(
                        loader=self.validation_loader, device=self.device, net=self.model,
                        num_classes=self._cfg['num_classes'], categories=self._cfg['categories'],
                        output_size=self._cfg['original_size'],
                        labels_size=self._cfg['encoder_size'],
                        selector=self._cfg['selector'],
                        classes=self._cfg['eval_classes'],
                        mixed_precision=self._cfg['mixed_precision'],
                        encoder_only=self._cfg['encoder_only'])
                    if is_main_process():
                        self.writer.add_scalar('test pixel accuracy',
                                               test_pixel_accuracy,
                                               current_step_num)
                        self.writer.add_scalar('test mIoU',
                                               test_mIoU,
                                               current_step_num)
                    self.model.train()

                    # Record best model (straight to disk)
                    if test_mIoU > best_mIoU:
                        best_mIoU = test_mIoU
                        save_checkpoint(net=self.model.module if self._cfg['distributed'] else self.model,
                                        optimizer=None,
                                        lr_scheduler=None,
                                        filename=os.path.join(self._cfg['exp_dir'], 'model.pt'))

            # Evaluate training accuracies (same metric as validation, but must be on-the-fly to save time)
            conf_mat.reduce_from_all_processes()
            acc_global, acc, iu = conf_mat.compute()
            print(self._cfg['categories'])
            print((
                'Pixel acc: {:.2f}\n'
                'Pixel acc (per-class): {}\n'
                'IoU (per-class): {}\n'
                'Mean IoU: {:.2f}').format(
                acc_global.item() * 100,
                ['{:.2f}'.format(i) for i in (acc * 100).tolist()],
                ['{:.2f}'.format(i) for i in (iu * 100).tolist()],
                iu.mean().item() * 100))

            train_pixel_acc = acc_global.item() * 100
            train_mIoU = iu.mean().item() * 100
            if is_main_process():
                self.writer.add_scalar('train pixel accuracy',
                                       train_pixel_acc,
                                       epoch + 1)
                self.writer.add_scalar('train mIoU',
                                       train_mIoU,
                                       epoch + 1)

            epoch += 1
            print('Epoch time: %.2fs' % (time.time() - time_now))

    def get_validation_dataset(self, cfg):
        if not self._cfg['validation']:
            return None
        validation_transforms = TRANSFORMS.from_dict(cfg['test_augmentation'])
        validation_set = DATASETS.from_dict(cfg['dataset'],
                                            image_set='val',
                                            transforms=validation_transforms)

        return validation_set

    def clean(self):
        super().clean()
        if is_main_process():
            print('Segmentation models used to be evaluated upon training, now please run a separate --val for eval!')
