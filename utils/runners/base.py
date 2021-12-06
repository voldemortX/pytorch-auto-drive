# Define every component in one line
# cfg: config file, pure dict
# args: command line args from argparse
import torch
from torch.utils.tensorboard import SummaryWriter
from abc import ABC, abstractmethod

from ..datasets import DATASETS, dict_collate_fn
from ..losses import LOSSES
from ..lr_schedulers import LR_SCHEDULERS
from ..models import MODELS
from ..optimizers import OPTIMIZERS
from ..transforms import TRANSFORMS
from ..ddp_utils import init_distributed_mode, is_main_process
from ..common import load_checkpoint


def get_collate_fn(name):
    if name is not None and name == 'dict_collate_fn':
        return dict_collate_fn
    else:
        return None


def get_sampler(ddp, dataset):
    if ddp:
        sampler = torch.utils.data.distributed.DistributedSampler(dataset)
    else:
        sampler = torch.utils.data.RandomSampler(dataset)

    return sampler


class BaseRunner(ABC):
    def __init__(self, cfg):
        self.model = MODELS.from_dict(cfg['model'])

    @abstractmethod
    def run(self, *args, **kwargs):
        pass

    def get_device_and_move_model(self, *args, **kwargs):
        device = torch.device('cpu')
        if torch.cuda.is_available():
            device = torch.device('cuda:0')
        self.model.to(device)

        return device

    def load_checkpoint(self, ckpt_filename):
        # [Possible BC-Break] Get rid of scheduler and optimizer loading
        if ckpt_filename is not None:
            load_checkpoint(net=self.model, lr_scheduler=None, optimizer=None, filename=ckpt_filename)

    @staticmethod
    def update_cfg(cfg, updates):
        if not isinstance(updates, dict):
            updates = vars(updates)
        return cfg.update(updates)


class BaseTrainer(BaseRunner):
    def __init__(self, cfg, args):
        super().__init__(cfg)
        net_without_ddp, self.device = self.get_device_and_move_model(args)
        self._cfg = cfg['train']
        self.update_cfg(self._cfg, args)
        self.writer = self.get_writer()
        self.load_checkpoint(self._cfg['continue_from'])

        # Dataset
        self.collate_fn = get_collate_fn(self._cfg['collate_fn'])
        transforms = TRANSFORMS.from_dict(cfg['train_augmentation'])
        dataset = DATASETS.from_dict(cfg['dataset'],
                                     transforms=transforms)
        self.train_sampler = get_sampler(self._cfg['ddp'], dataset)
        self.dataloader = torch.utils.data.DataLoader(dataset=dataset,
                                                      batch_size=self._cfg['batch_size'],
                                                      collate_fn=self.collate_fn,
                                                      sampler=self.train_sampler,
                                                      num_workers=self._cfg['workers'])
        self.validation_loader = self.get_validation_dataset(cfg)

        # Optimizer, LR scheduler, etc.
        self.optimizer = OPTIMIZERS.from_dict(cfg['optimizer'],
                                              net_without_ddp=net_without_ddp)
        self.lr_scheduler = LR_SCHEDULERS.from_dict(cfg['lr_scheduler'],
                                                    optimizer=self.optimizer)
        self.criterion = LOSSES.from_dict(cfg['loss'])

    def get_device_and_move_model(self, args):
        init_distributed_mode(args)
        device = torch.device(args.device)
        print(device)
        self.model.to(device)

        if args.distributed:
            self.model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.model)
        net_without_ddp = self.model
        if args.distributed:
            self.model = torch.nn.parallel.DistributedDataParallel(self.model,
                                                                   device_ids=[args.gpu],
                                                                   find_unused_parameters=True)
            net_without_ddp = self.model.module

        return net_without_ddp, device

    def get_writer(self):
        return SummaryWriter('runs/' + self._cfg['exp_name']) if is_main_process() else None

    @abstractmethod
    def run(self, *args, **kwargs):
        pass

    def get_validation_dataset(self, *args, **kwargs):
        return None


class BaseTester(BaseRunner):
    image_sets = ['val']

    def __init__(self, cfg, args):
        super().__init__(cfg)
        self._cfg = cfg['test']
        self.update_cfg(self._cfg, args)
        self.device = self.get_device_and_move_model()
        self.load_checkpoint(self._cfg['continue_from'])

        # Dataset
        transforms = TRANSFORMS.from_dict(cfg['test_augmentation'])
        dataset = DATASETS.from_dict(cfg['dataset'],
                                     image_set=self.image_sets[self._cfg['state'] - 1],
                                     transforms=transforms)

        # Dataloader
        collate_fn = get_collate_fn(self._cfg['collate_fn'])
        self.dataloader = torch.utils.data.DataLoader(dataset=dataset,
                                                      batch_size=self._cfg['batch_size'],
                                                      collate_fn=collate_fn,
                                                      num_workers=self._cfg['workers'],
                                                      shuffle=False)

    @abstractmethod
    def run(self, *args, **kwargs):
        pass
