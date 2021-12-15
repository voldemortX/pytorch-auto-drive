import os
import torch
import time
if torch.__version__ >= '1.6.0':
    from torch.cuda.amp import autocast, GradScaler
else:
    from ..torch_amp_dummy import autocast, GradScaler

from ..common import save_checkpoint
from ..ddp_utils import reduce_dict, is_main_process
from .lane_det_tester import LaneDetTester
from .base import BaseTrainer, DATASETS, TRANSFORMS


class LaneDetTrainer(BaseTrainer):
    def __init__(self, cfg, args):
        super().__init__(cfg, args)

    def run(self):
        # Should be the same as segmentation, given customized loss classes
        self.model.train()
        epoch = 0
        running_loss = None  # Dict logging for every loss (too many losses in this task)
        loss_num_steps = int(len(self.dataloader) / 10) if len(self.dataloader) > 10 else 1
        if self._cfg['mixed_precision']:
            scaler = GradScaler()

        # Training
        best_validation = 0
        while epoch < self._cfg['num_epochs']:
            self.model.train()
            if self._cfg['distributed']:
                self.train_sampler.set_epoch(epoch)
            time_now = time.time()
            for i, data in enumerate(self.dataloader, 0):
                if self._cfg['seg']:
                    inputs, labels, existence = data
                    inputs, labels, existence = inputs.to(self.device), labels.to(self.device), existence.to(self.device)
                else:
                    inputs, labels = data
                    inputs = inputs.to(self.device)
                    labels = [{k: v.to(self.device) for k, v in label.items()} for label in labels]  # Seems slow
                self.optimizer.zero_grad()

                with autocast(self._cfg['mixed_precision']):
                    # To support intermediate losses for SAD
                    if self._cfg['seg']:
                        loss, log_dict = self.criterion(inputs, labels, existence,
                                                        self.model, self._cfg['input_size'])
                    else:
                        loss, log_dict = self.criterion(inputs, labels,
                                                        self.model)

                if self._cfg['mixed_precision']:
                    scaler.scale(loss).backward()
                    scaler.step(self.optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    self.optimizer.step()

                self.lr_scheduler.step()

                log_dict = reduce_dict(log_dict)
                if running_loss is None:  # Because different methods may have different values to log
                    running_loss = {k: 0.0 for k in log_dict.keys()}
                for k in log_dict.keys():
                    running_loss[k] += log_dict[k]
                current_step_num = int(epoch * len(self.dataloader) + i + 1)

                # Record losses
                if current_step_num % loss_num_steps == (loss_num_steps - 1):
                    for k in running_loss.keys():
                        print('[%d, %d] %s: %.4f' % (epoch + 1, i + 1, k, running_loss[k] / loss_num_steps))
                        # Logging only once
                        if is_main_process():
                            self.writer.add_scalar(k, running_loss[k] / loss_num_steps, current_step_num)
                        running_loss[k] = 0.0

                # Record checkpoints
                if self._cfg['validation']:
                    if current_step_num % self._cfg['val_num_steps'] == (self._cfg['val_num_steps'] - 1) or \
                            current_step_num == self._cfg['num_epochs'] * len(self.dataloader):
                        test_pixel_accuracy, test_mIoU = LaneDetTester.fast_evaluate(
                            loader=self.validation_loader,
                            device=self.device,
                            net=self.model,
                            num_classes=self._cfg['num_classes'],
                            output_size=self._cfg['input_size'],
                            mixed_precision=self._cfg['mixed_precision'])
                        if is_main_process():
                            self.writer.add_scalar('test pixel accuracy',
                                                   test_pixel_accuracy,
                                                   current_step_num)
                            self.writer.add_scalar('test mIoU',
                                                   test_mIoU,
                                                   current_step_num)
                        self.model.train()

                        # Record best model (straight to disk)
                        if test_mIoU > best_validation:
                            best_validation = test_mIoU
                            save_checkpoint(net=self.model.module if self._cfg['distributed'] else self.model,
                                            optimizer=None,
                                            lr_scheduler=None,
                                            filename=os.path.join(self._cfg['exp_dir'] + 'model.pt'))

            epoch += 1
            print('Epoch time: %.2fs' % (time.time() - time_now))

        # For no-evaluation mode
        if not self._cfg['validation']:
            save_checkpoint(net=self.model.module if self._cfg['distributed'] else self.model,
                            optimizer=None,
                            lr_scheduler=None,
                            filename=os.path.join(self._cfg['exp_dir'] + 'model.pt'))

    def get_validation_dataset(self, cfg):
        if not self._cfg['validation']:
            return None
        validation_transforms = TRANSFORMS.from_dict(cfg['test_augmentation'])
        validation_set = DATASETS.from_dict(cfg['validation_dataset'],
                                            transforms=validation_transforms)
        return validation_set
