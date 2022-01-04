import argparse
import torch

from importmagician import import_from
with import_from('./'):
    from utils.args import read_config, parse_arg_cfg, cmd_dict, add_shortcuts
    from utils.runners import LaneDetDir
    try:
        from utils.common import warnings
    except ImportError:
        import warnings

if __name__ == '__main__':
    # Settings
    parser = argparse.ArgumentParser(description='PytorchAutoDrive lane directory vis', conflict_handler='resolve')
    add_shortcuts(parser)

    parser.add_argument('--config', type=str, help='Path to config file', required=True)
    parser.add_argument('--pred', action='store_true',
                        help='Whether to predict from a model')
    parser.add_argument('--image-path', type=str, required=True,
                        help='Image input path')
    parser.add_argument('--save-path', type=str, required=True,
                        help='Result output path')

    # Optional args/to overwrite configs
    parser.add_argument('--mask-path', type=str,
                        help='Mask input path, if both mask & keypoint are None,'
                             'inference will be performed ')
    parser.add_argument('--keypoint-path', type=str,
                        help='Keypoint input path (expect json/txt file in CULane format, [x, y]),'
                             'if both mask & keypoint are None, inference will be performed')
    parser.add_argument('--image-suffix', type=str, default='.jpg',
                        help='Image file suffix')
    parser.add_argument('--keypoint-suffix', type=str, default='.txt',
                        help='Keypoint file suffix')
    parser.add_argument('--mask-suffix', type=str, default='.png',
                        help='Segmentation mask file suffix')
    parser.add_argument('--mixed-precision', action='store_true',
                        help='Enable mixed precision training')
    parser.add_argument('--cfg-options', type=cmd_dict,
                        help='Override config options with \"x1=y1 x2=y2 xn=yn\"')

    group2 = parser.add_mutually_exclusive_group()
    group2.add_argument('--continue-from', type=str,
                        help='[Deprecated] Continue training from a previous checkpoint')
    group2.add_argument('--checkpoint', type=str,
                        help='Continue/Load from a previous checkpoint')

    retain_args = ['mixed_precision', 'pred',
                   'image_path', 'save_path', 'mask_path', 'keypoint_path',
                   'image_suffix', 'keypoint_suffix', 'mask_suffix']

    args = parser.parse_args()

    # Parse configs and build model
    if args.mixed_precision and torch.__version__ < '1.6.0':
        warnings.warn('PyTorch version too low, mixed precision training is not available.')
    cfg = read_config(args.config)
    args, cfg = parse_arg_cfg(args, cfg)

    cfg_runner_key = 'vis' if 'vis' in cfg.keys() else 'test'
    for k in retain_args:
        cfg[cfg_runner_key][k] = vars(args)[k]
    if not cfg['pred']:
        cfg['model'] = None
    runner = LaneDetDir(cfg=cfg)
    runner.run()
    runner.clean()
