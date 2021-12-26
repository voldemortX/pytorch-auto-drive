# Define a series of shortcuts for commandline use of main_*.py
# a_b equals --a-b in commandline

SHORTCUTS = dict(
    exp_name=dict(keys=['train.exp_name', 'test.exp_name'], type=str,
                  help='Name of experiment'),
    checkpoint=dict(keys=['train.checkpoint', 'test.checkpoint'], type=str,
                    help='Continue/Load from a previous checkpoint'),
    device=dict(keys=['train.device', 'test.device'], type=str,
                help='CPU is not recommended!'),
    workers=dict(keys=['train.workers', 'test.workers'], type=int,
                 help='Number of workers (threads) when loading data.'
                      'Recommend value for training=~ batch size'),
    batch_size=dict(keys=['train.batch_size', 'test.batch_size'], type=int,
                    help='input batch size. Recommend 4 times the training batch size in testing'),
    save_dir=dict(keys=['train.save_dir', 'test.save_dir'], type=str,
                  help='Path prefix to save all files excluding tensorboard log.'),

    val_num_steps=dict(keys=['train.val_num_steps'], type=int,
                       help='Validation frequency'),
    world_size=dict(keys=['train.world_size'], type=int,
                    help='Number of distributed processes'),
    dist_url=dict(keys=['train.dist_url'], type=str,
                  help='url used to set up distributed training'),

    thresh=dict(keys=['test.thresh'], type=float,
                help='Threshold for detection tasks.'),

    lr=dict(keys=['optimizer.lr'], type=float,
            help='Learning rate'),
    weight_decay=dict(keys=['optimizer.weight_decay'], type=float,
                      help='Weight decay'),

    warmup_steps=dict(keys=['lr_scheduler.warmup_steps'], type=int,
                      help='Learning rate warmup steps.'),
    epochs=dict(keys=['lr_scheduler.epochs', 'train.num_epochs'], type=int,
                help='Number of epochs')
)

DEPRECATION_MAP = dict(
    continue_from=dict(valid='checkpoint', message=''),
    do_not_save=dict(valid=None, message='Please delete the .pt files yourself!'),
    method=dict(valid=None, message='Please use the config files to define models!'),
    model=dict(valid=None, message='Please use the config files to define models!'),
    backbone=dict(valid=None, message='Please use the config files to define models!')
)
