import argparse
import torch

from importmagician import import_from
with import_from('./'):
    from utils.args import read_config, parse_arg_cfg, cmd_dict, add_shortcuts
    from utils.models import MODELS
    from utils.common import load_checkpoint
    from utils.profiling_utils import init_dataset, speed_evaluate_real, speed_evaluate_simple, model_profile

if __name__ == '__main__':
    # Settings
    parser = argparse.ArgumentParser(description='PytorchAutoDrive Profiling', conflict_handler='resolve')
    add_shortcuts(parser)

    parser.add_argument('--config', type=str, help='Path to config file', required=True)

    # Optional args/to overwrite configs
    parser.add_argument('--height', type=int, default=288,
                        help='Image input height (default: 288)')
    parser.add_argument('--width', type=int, default=800,
                        help='Image input width (default: 800)')
    parser.add_argument('--mode', type=str, default='simple',
                        help='Profiling mode (simple/real)')
    parser.add_argument('--times', type=int, default=1,
                        help='Select test times')
    parser.add_argument('--cfg-options', type=cmd_dict,
                        help='Override config options with \"x1=y1 x2=y2 xn=yn\", tuple value not supported, try list')

    group2 = parser.add_mutually_exclusive_group()
    group2.add_argument('--continue-from', type=str,
                        help='[Deprecated] Continue training from a previous checkpoint')
    group2.add_argument('--checkpoint', type=str,
                        help='Continue/Load from a previous checkpoint')

    args = parser.parse_args()

    # Parse configs and build model
    cfg = read_config(args.config)
    args, cfg = parse_arg_cfg(args, cfg)
    net = MODELS.from_dict(cfg['model'])

    device = torch.device('cpu')
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    print(device)

    net.to(device)

    macs, _ = model_profile(net, args.height, args.width, device)
    params = sum(p.numel() for p in net.parameters())
    print('FLOPs(G): {: .2f}'.format(2 * macs / 1e9))
    print('Number of parameters: {: .2f}'.format(params / 1e6))
    print('Profiling, please clear your GPU memory before doing this.')
    if args.mode == 'simple':
        dummy = torch.ones((1, 3, args.height, args.width))
        print(dummy.dtype)
        fps = []
        for i in range(0, args.times):
            fps.append(speed_evaluate_simple(net=net, device=device, dummy=dummy, num=300))
        print('GPU FPS: {: .2f}'.format(max(fps)))
    elif args.mode == 'real':
        if cfg['test']['checkpoint'] is not None:
            load_checkpoint(net=net, optimizer=None, lr_scheduler=None, filename=cfg['test']['checkpoint'])
        val_loader = init_dataset(cfg['dataset'], cfg['test_augmentations'], (args.height, args.width))
        fps = []
        gpu_fps = []
        for i in range(0, args.times):
            fps_item, gpu_fps_item = speed_evaluate_real(net=net, device=device, loader=val_loader, num=300)
            fps.append(fps_item)
            gpu_fps.append(gpu_fps_item)
        print('Real FPS: {: .2f}'.format(max(fps)))
        print('GPU FPS: {: .2f}'.format(max(gpu_fps)))
    else:
        raise ValueError
