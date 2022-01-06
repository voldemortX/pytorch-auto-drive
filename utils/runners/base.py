# Define every component in one line
# cfg: config file, pure dict
# args: command line args from argparse
import os
import torch
from torch.utils.tensorboard import SummaryWriter
from abc import ABC, abstractmethod

try:
    import ujson as json
except ImportError:
    import json

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

    def clean(self, *args, **kwargs):
        # Cleanups and a hook for after-run messages/ops
        if hasattr(self, '_cfg') and 'exp_dir' in self._cfg.keys():
            print('Files saved at: {}.\nTensorboard log at: {}'.format(
                self._cfg['exp_dir'],
                os.path.join(self._cfg['save_dir'], 'tb_logs', self._cfg['exp_name'])
            ))

    def get_device_and_move_model(self, *args, **kwargs):
        device = torch.device('cpu')
        if torch.cuda.is_available():
            device = torch.device('cuda:0')
        print(device)
        self.model.to(device)

        return device

    def load_checkpoint(self, ckpt_filename):
        # [Possible BC-Break] Get rid of scheduler and optimizer loading
        if ckpt_filename is not None:
            load_checkpoint(net=self.model, lr_scheduler=None, optimizer=None, filename=ckpt_filename)

    def get_dataset_statics(self, dataset, map_dataset_statics, exist_ok=False):
        assert hasattr(self, '_cfg')
        if map_dataset_statics is not None:
            for k in map_dataset_statics:
                if exist_ok and k in self._cfg.keys():
                    continue
                if isinstance(dataset, str):
                    from utils import datasets
                    attr = getattr(datasets.__dict__[dataset], k)
                else:
                    attr = getattr(dataset, k)
                self._cfg[k] = attr

    def init_exp_dir(self, cfg, cfg_prefix=None):
        # Init work directory and save parsed configs for reference
        assert hasattr(self, '_cfg')
        exp_dir = os.path.join(self._cfg['save_dir'], self._cfg['exp_name'])
        os.makedirs(exp_dir, exist_ok=True)
        self._cfg['exp_dir'] = exp_dir
        with open(os.path.join(exp_dir, cfg_prefix + '_cfg.json'), 'w') as f:
            f.write(json.dumps(cfg, indent=4))

    @staticmethod
    def update_cfg(cfg, updates):
        # Update by argparse object/dict
        if not isinstance(updates, dict):
            updates = vars(updates)
        return cfg.update(updates)

    @staticmethod
    def write_mp_log(log_file, content, append=True):
        # Multi-processing log writing
        import fcntl
        with open(log_file, 'a' if append else 'w') as f:
            # Safe writing with locks
            fcntl.flock(f, fcntl.LOCK_EX)
            f.write(content)
            fcntl.flock(f, fcntl.LOCK_UN)


class BaseTrainer(BaseRunner):
    def __init__(self, cfg, map_dataset_statics=None):
        super().__init__(cfg)
        self._cfg = cfg['train']
        net_without_ddp, self.device = self.get_device_and_move_model()
        if 'val_num_steps' in self._cfg.keys():
            self._cfg['validation'] = self._cfg['val_num_steps'] > 0
        self.init_exp_dir(cfg, 'train')
        self.writer = self.get_writer()
        self.load_checkpoint(self._cfg['checkpoint'])

        # Dataset
        self.collate_fn = get_collate_fn(self._cfg['collate_fn'])
        transforms = TRANSFORMS.from_dict(cfg['train_augmentation'])
        dataset = DATASETS.from_dict(cfg['dataset'],
                                     transforms=transforms)
        self.get_dataset_statics(dataset, map_dataset_statics)
        self.train_sampler = get_sampler(self._cfg['distributed'], dataset)
        self.dataloader = torch.utils.data.DataLoader(dataset=dataset,
                                                      batch_size=self._cfg['batch_size'],
                                                      collate_fn=self.collate_fn,
                                                      sampler=self.train_sampler,
                                                      num_workers=self._cfg['workers'])
        validation_set = self.get_validation_dataset(cfg)
        self.validation_loader = None
        if validation_set is not None:
            val_bs = self._cfg.get('val_batch_size')
            if val_bs is None:
                val_bs = self._cfg['batch_size']
            self.validation_loader = torch.utils.data.DataLoader(dataset=validation_set,
                                                                 batch_size=val_bs,
                                                                 num_workers=self._cfg['workers'],
                                                                 shuffle=False,
                                                                 collate_fn=self.collate_fn)


        # Optimizer, LR scheduler, etc.
        self.optimizer = self.get_optimizer(cfg['optimizer'], net_without_ddp)
        self.lr_scheduler = LR_SCHEDULERS.from_dict(cfg['lr_scheduler'],
                                                    optimizer=self.optimizer,
                                                    len_loader=len(self.dataloader))
        self.criterion = LOSSES.from_dict(cfg['loss'])

    def get_device_and_move_model(self):
        init_distributed_mode(self._cfg)
        device = torch.device(self._cfg['device'])
        print(device)
        self.model.to(device)

        if self._cfg['distributed']:
            self.model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.model)
        net_without_ddp = self.model
        if self._cfg['distributed']:
            self.model = torch.nn.parallel.DistributedDataParallel(self.model,
                                                                   device_ids=[self._cfg['gpu']],
                                                                   find_unused_parameters=True)
            net_without_ddp = self.model.module

        return net_without_ddp, device

    def get_writer(self):
        return SummaryWriter(os.path.join(self._cfg['save_dir'],
                                          'tb_logs',
                                          self._cfg['exp_name'])) if is_main_process() else None

    @staticmethod
    def get_optimizer(optimizer_cfg, net):
        parameters = optimizer_cfg.pop('parameters') if 'parameters' in optimizer_cfg.keys() else None
        if parameters is None:  # For BC
            parameters = net.parameters()
        else:  # replace str with actual parameter groups
            group_keys = [d['params'] for d in parameters if d['params'] != '__others__']
            for i in range(len(parameters)):
                assert type(parameters[i]['params']) == str, 'Use string as placeholder in your config!'
                if parameters[i]['params'] == '__others__':
                    other_params = [v for _, v in list(filter(lambda kv: all([group_key not in kv[0]
                                                                              for group_key in group_keys]),
                                                              net.named_parameters()))]
                    parameters[i] = {'params': other_params}
                else:
                    parameters[i]['params'] = [v for _, v in list(filter(lambda kv: parameters[i]['params'] in kv[0],
                                                                         net.named_parameters()))]

        return OPTIMIZERS.from_dict(optimizer_cfg, parameters=parameters)

    @abstractmethod
    def run(self, *args, **kwargs):
        pass

    def get_validation_dataset(self, *args, **kwargs):
        return None

    def clean(self):
        super().clean()
        if self.writer is not None:
            self.writer.close()


class BaseTester(BaseRunner):
    image_sets = ['val']

    def __init__(self, cfg, map_dataset_statics=None):
        super().__init__(cfg)
        self._cfg = cfg['test']
        self.init_exp_dir(cfg, self.image_sets[self._cfg['state'] - 1])
        self.device = self.get_device_and_move_model()
        self.load_checkpoint(self._cfg['checkpoint'])

        # Dataset
        transforms = TRANSFORMS.from_dict(cfg['test_augmentation'])
        dataset = DATASETS.from_dict(cfg['test_dataset'] if 'test_dataset' in cfg.keys() else cfg['dataset'],
                                     image_set=self.image_sets[self._cfg['state'] - 1],
                                     transforms=transforms)
        self.get_dataset_statics(dataset, map_dataset_statics)

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


class BaseVisualizer(BaseRunner):
    dataset_tensor_statistics = []
    dataset_statistics = []

    def __init__(self, cfg):
        super().__init__(cfg)
        self._cfg = cfg['vis'] if 'vis' in cfg.keys() else cfg['test']
        self.dataloader, dataset = self.get_loader(cfg)
        self.get_dataset_statics(dataset, set(self.dataset_statistics).union(set(self.dataset_tensor_statistics)),
                                 exist_ok=True)
        for k in self.dataset_tensor_statistics:
            self._cfg[k] = torch.tensor(self._cfg[k])
        if self._cfg['pred']:
            self.device = self.get_device_and_move_model()
            self.load_checkpoint(self._cfg['checkpoint'])
            for k in self.dataset_tensor_statistics:
                self._cfg[k] = self._cfg[k].to(self.device)

    @abstractmethod
    def run(self, *args, **kwargs):
        pass

    @abstractmethod
    def get_loader(self, *args, **kwargs):
        pass
