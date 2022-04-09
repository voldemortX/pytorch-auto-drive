import argparse
import torch

from importmagician import import_from
with import_from('./'):
    from utils.args import read_config, parse_arg_cfg, cmd_dict, add_shortcuts
    from utils.runners import LaneDetVideo
    try:
        from utils.common import warnings
    except ImportError:
        import warnings

if __name__ == '__main__':
    # Settings
    parser = argparse.ArgumentParser(description='PytorchAutoDrive lane video vis', conflict_handler='resolve')
    add_shortcuts(parser)

    parser.add_argument('--config', type=str, help='Path to config file', required=True)
    parser.add_argument('--video-path', type=str, required=True,
                        help='Video input path')
    parser.add_argument('--save-path', type=str, required=True,
                        help='Result output path')
    parser.add_argument('--style', type=str, default='point',
                        help='Lane visualization style: point/line/bezier')

    # Optional args/to overwrite configs
    parser.add_argument('--mixed-precision', action='store_true',
                        help='Enable mixed precision training')
    parser.add_argument('--cfg-options', type=cmd_dict,
                        help='Override config options with \"x1=y1 x2=y2 xn=yn\"')

    group2 = parser.add_mutually_exclusive_group()
    group2.add_argument('--continue-from', type=str,
                        help='[Deprecated] Continue training from a previous checkpoint')
    group2.add_argument('--checkpoint', type=str,
                        help='Continue/Load from a previous checkpoint')

    retain_args = ['mixed_precision',
                   'video_path', 'save_path', 'style']

    args = parser.parse_args()

    # Parse configs and build model
    if args.mixed_precision and torch.__version__ < '1.6.0':
        warnings.warn('PyTorch version too low, mixed precision training is not available.')
    cfg = read_config(args.config)
    args, cfg = parse_arg_cfg(args, cfg)

    cfg_runner_key = 'vis' if 'vis' in cfg.keys() else 'test'
    for k in retain_args:
        cfg[cfg_runner_key][k] = vars(args)[k]
    cfg[cfg_runner_key]['pred'] = True  # Must predict
    runner = LaneDetVideo(cfg=cfg)
    runner.run()
    runner.clean()
