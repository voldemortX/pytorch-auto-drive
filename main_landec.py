import torch
if torch.backends.cudnn.version() < 8000:
    torch.backends.cudnn.benchmark = True
# torch.multiprocessing.set_sharing_strategy('file_system')
import resource
import argparse

from .utils.args import parse_arg_cfg
from .utils.runners import LaneDetTrainer, LaneDetTester


if __name__ == '__main__':
    # ulimit
    rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (8192, rlimit[1]))

    # Settings (user input > config > argparse defaults)
    parser = argparse.ArgumentParser(description='PyTorch Auto-drive')
    parser.add_argument('--exp-name', type=str,
                        help='Name of experiment')
    parser.add_argument('--workers', type=int,
                        help='Number of workers (threads) when loading data.'
                             'Recommend value for training: batch_size / 2')
    parser.add_argument('--batch-size', type=int,
                        help='input batch size. Recommend 4 times the training batch size in testing')
    parser.add_argument('--mixed-precision', type=bool,
                        help='Enable mixed precision training')
    parser.add_argument('--continue-from', type=str,
                        help='Continue training from a previous checkpoint')
    parser.add_argument('--state', type=int,
                        help='Conduct validation(3)/final test(2)/fast validation(1)/normal training(0)')
    parser.add_argument('--world-size', type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist-url', type=str, help='url used to set up distributed training')
    parser.add_argument('--device', type=str, help='CPU is not recommended!')
    parser.add_argument('--config', type=str, help='Path to config file')
    args = parser.parse_args()
    
