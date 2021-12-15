import time
import torch
import argparse
try:
    from utils.common import warnings
except ImportError:
    import warnings
if torch.backends.cudnn.version() < 8000:
    torch.backends.cudnn.benchmark = True

from utils.args import parse_arg_cfg, read_config, map_states
from utils.runners import SegTrainer, SegTester


if __name__ == '__main__':
    # Settings (user input > config > argparse defaults)
    parser = argparse.ArgumentParser(description='PytorchAutoDrive Semantic Segmentation')

    # Required args
    parser.add_argument('--config', type=str, help='Path to config file', required=True)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--train', action='store_true')
    group.add_argument('--val', action='store_true')  # There is no test labels available for these datasets
    group.add_argument('--state', type=int,
                       help='[Deprecated] validation set testing(1)/training(0)')

    # Optional args/to overwrite configs
    group2 = parser.add_mutually_exclusive_group()
    group2.add_argument('--continue-from', type=str,
                        help='[Deprecated] Continue training from a previous checkpoint')
    group2.add_argument('--checkpoint', type=str,
                        help='Continue/Load from a previous checkpoint')

    parser.add_argument('--exp-name', type=str,
                        help='Name of experiment')
    parser.add_argument('--workers', type=int,
                        help='Number of workers (threads) when loading data.'
                             'Recommend value for training: batch_size')
    parser.add_argument('--batch-size', type=int,
                        help='input batch size. Recommend 4 times the training batch size in testing')
    parser.add_argument('--mixed-precision', action='store_true',
                        help='Enable mixed precision training')
    parser.add_argument('--val-num-steps', type=int, help='Validation frequency')
    parser.add_argument('--world-size', type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist-url', type=str, help='url used to set up distributed training')
    parser.add_argument('--device', type=str, help='CPU is not recommended!')
    parser.add_argument('--save-dir', type=str, help='Path prefix to save all files excluding tensorboard log.')
    parser.add_argument('--do-not-save', action='store_true',
                        help='[Deprecated] Please delete .pt files by yourself.')

    # Deprecations, Defaults and such
    defaults = {
        'exp_name': time.time(),
        'workers': 8,
        'batch_size': 8,
        'mixed_precision': False,
        'checkpoint': None,
        'continue_from': None,
        'val_num_steps': 1000,
        'world_size': 0,
        'dist_url': 'env://',
        'device': 'cuda',
        'save_dir': '',
        'do_not_save': False
    }
    states = ['train', 'val']
    deprecation_map = {
        'continue_from': {'valid': 'checkpoint', 'message': ''},
        'do_not_save': {'valid': None, 'message': 'Please delete the .pt files yourself!', 'expected': False}
    }

    args = parser.parse_args()
    if args.state is not None:
        warnings.warn('--state={} is deprecated, it is recommended to specify with --{}'.format(
            args.state, states[args.state]))
    args.state = map_states(args, states)
    if args.mixed_precision and torch.__version__ < '1.6.0':
        warnings.warn('PyTorch version too low, mixed precision training is not available.')

    # Parse configs and execute runner
    cfg = read_config(args.config)
    cfg_runner_key = 'train' if args.state == 0 else 'test'
    Runner = SegTrainer if args.state == 0 else SegTester
    args, cfg[cfg_runner_key] = parse_arg_cfg(args, cfg[cfg_runner_key],
                                              defaults, required=['state', 'config'], deprecation_map=deprecation_map)
    runner = Runner(cfg=cfg, args=args)
    runner.run()
    runner.clean()
