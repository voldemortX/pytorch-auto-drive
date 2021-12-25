# Define a series of shortcuts for commandline use of main_*.py

SHORTCUTS = {
    '--exp-name': {'keys': ['train.exp_name', 'test.exp_name'], 'type': str,
                   'help': 'Name of experiment'},
    '--checkpoint': {'keys': ['train.checkpoint', 'test.checkpoint'], 'type': str,
                     'help': 'Continue/Load from a previous checkpoint'},
    '--device': {'keys': ['train.device', 'test.device'], 'type': str,
                 'help': 'CPU is not recommended!'},
    '--workers': {'keys': ['train.workers', 'test.workers'], 'type': int,
                  'help': 'Number of workers (threads) when loading data.'
                          'Recommend value for training: ~ batch size'},
    '--batch-size': {'keys': ['train.batch_size', 'test.batch_size'], 'type': int,
                     'help': 'input batch size. Recommend 4 times the training batch size in testing'},
    '--save-dir': {'keys': ['train.save_dir', 'test.save_dir'], 'type': str,
                   'help': 'Path prefix to save all files excluding tensorboard log.'},

    '--val-num-steps': {'keys': ['train.val_num_steps'], 'type': int,
                        'help': 'Validation frequency'},
    '--world-size': {'keys': ['train.world_size'], 'type': int,
                     'help': 'Number of distributed processes'},
    '--dist-url': {'keys': ['train.dist_url'], 'type': str,
                   'help': 'url used to set up distributed training'},

    '--thresh': {'keys': ['test.thresh'], 'type': float,
                 'help': 'Threshold for detection tasks.'},

    '--lr': {'keys': ['optimizer.lr'], 'type': float,
             'help': 'Learning rate'},
    '--weight-decay': {'keys': ['optimizer.weight_decay'], 'type': float,
                       'help': 'Weight decay'},

    '--warmup-steps': {'keys': ['lr_scheduler.warmup_steps'], 'type': int,
                       'help': 'Learning rate warmup steps.'},
    '--epochs': {'keys': ['lr_scheduler.epochs', 'train.num_epochs'], 'type': int,
                 'help': 'Number of epochs'}
}

DEPRECATION_MAP = {
    'continue_from': {'valid': 'checkpoint', 'message': ''},
    'do_not_save': {'valid': None, 'message': 'Please delete the .pt files yourself!'},
    'method': {'valid': None, 'message': 'Please use the config files to define models!'},
    'model': {'valid': None, 'message': 'Please use the config files to define models!'},
    'backbone': {'valid': None, 'message': 'Please use the config files to define models!'}
}
