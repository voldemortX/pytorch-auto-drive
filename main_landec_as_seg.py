import time
import torch
import argparse
import yaml
from torch.utils.tensorboard import SummaryWriter
from utils.losses import LaneLoss, SADLoss
from utils.all_utils_semseg import load_checkpoint
from utils.all_utils_landec_as_seg import init, train_schedule, test_one_set, erfnet_tusimple, erfnet_culane, \
    fast_evaluate, vgg16_culane

if __name__ == '__main__':
    # Settings
    parser = argparse.ArgumentParser(description='PyTorch Auto-drive')
    parser.add_argument('--exp-name', type=str, default='',
                        help='Name of experiment')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='Initial learning rate (default: 0.01)')
    parser.add_argument('--epochs', type=int, default=30,
                        help='Number of epochs (default: 30)')
    parser.add_argument('--val-num-steps', type=int, default=0,
                        help='Validation frequency (default: 0), 0: no online evaluation')
    parser.add_argument('--dataset', type=str, default='tusimple',
                        help='Train/Evaluate on TuSimple (voc) / CULane (culane) (default: tusimple)')
    parser.add_argument('--method', type=str, default='baseline',
                        help='method selection (scnn/sad/baseline/etc) (default: scnn)')
    parser.add_argument('--backbone', type=str, default='erfnet',
                        help='backbone selection (erfnet/vgg16) (default: erfnet)')
    parser.add_argument('--batch-size', type=int, default=8,
                        help='input batch size (default: 8)')
    parser.add_argument('--mixed-precision', action='store_true', default=False,
                        help='Enable mixed precision training (default: False)')
    parser.add_argument('--continue-from', type=str, default=None,
                        help='Continue training from a previous checkpoint')
    parser.add_argument('--state', type=int, default=0,
                        help='Conduct validation(3)/final test(2)/fast validation(1)/normal training(0) (default: 0)')
    args = parser.parse_args()
    exp_name = str(time.time()) if args.exp_name == '' else args.exp_name
    states = ['train', 'valfast', 'test', 'val']
    with open(exp_name + '_' + states[args.state] + '_cfg.txt', 'w') as f:
        f.write(str(vars(args)))
    with open('configs.yaml', 'r') as f:  # Safer and cleaner than box/EasyDict
        configs = yaml.load(f, Loader=yaml.Loader)

    # Basic configurations
    mean = configs['GENERAL']['MEAN']
    std = configs['GENERAL']['STD']
    if args.dataset not in configs['LANE_DATASETS'].keys():
        raise ValueError
    num_classes = configs[configs['LANE_DATASETS'][args.dataset]]['NUM_CLASSES']
    input_sizes = configs[configs['LANE_DATASETS'][args.dataset]]['SIZES']
    gap = configs[configs['LANE_DATASETS'][args.dataset]]['GAP']
    ppl = configs[configs['LANE_DATASETS'][args.dataset]]['PPL']
    thresh = configs[configs['LANE_DATASETS'][args.dataset]]['THRESHOLD']
    weights = configs[configs['LANE_DATASETS'][args.dataset]]['WEIGHTS']
    base = configs[configs['LANE_DATASETS'][args.dataset]]['BASE_DIR']
    device = torch.device('cpu')
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    scnn = True if args.method == 'scnn' else False
    weights = torch.tensor(weights).to(device)
    if args.dataset == 'tusimple' and args.backbone == 'erfnet':
        net = erfnet_tusimple(num_classes=num_classes, scnn=scnn)
    elif args.dataset == 'culane'and args.backbone == 'erfnet':
        net = erfnet_culane(num_classes=num_classes, scnn=scnn)
    elif args.dataset == 'culane' and args.backbone == 'vgg16':
        net = net = vgg16_culane(num_classes=num_classes, scnn=scnn)
    else:
        raise ValueError
    print(device)
    net.to(device)
    # if args.model == 'scnn':
    #     # Gradient too large after spatial conv
    #     optimizer = torch.optim.SGD([
    #         {'params': net.encoder.parameters(), 'lr': 0.1 * args.lr},
    #         {'params': net.spatial_conv.parameters()},
    #         {'params': net.decoder.parameters()},
    #         {'params': net.aux_head.parameters()},
    #     ], lr=args.lr, momentum=0.9, weight_decay=1e-4)
    # else:
    optimizer = torch.optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)
    # optimizer = torch.optim.Adam(net.parameters(), lr=args.lr, betas=(0.9, 0.999),  eps=1e-08, weight_decay=1e-4)

    # Testing
    if args.state == 1 or args.state == 2 or args.state == 3:
        data_loader = init(batch_size=args.batch_size, state=args.state, dataset=args.dataset, input_sizes=input_sizes,
                           mean=mean, std=std, base=base)
        load_checkpoint(net=net, optimizer=None, lr_scheduler=None, filename=args.continue_from)
        if args.state == 1:  # Validate with mean IoU
            _, x = fast_evaluate(loader=data_loader, device=device, net=net,
                                 num_classes=num_classes, output_size=input_sizes[0],
                                 is_mixed_precision=args.mixed_precision)
            with open('log.txt', 'a') as f:
                f.write(exp_name + ' validation: ' + str(x) + '\n')

        else:  # Test with official scripts later (so just predict lanes here)
            test_one_set(net=net, device=device, loader=data_loader, is_mixed_precision=args.mixed_precision,
                         input_sizes=input_sizes, gap=gap, ppl=ppl, thresh=thresh, dataset=args.dataset)
    else:
        if args.method == 'scnn' or args.method == 'baseline':
            criterion = LaneLoss(weight=weights, ignore_index=255)
        elif args.method == 'sad':
            criterion = SADLoss()
        else:
            raise NotImplementedError

        writer = SummaryWriter('runs/' + exp_name)
        data_loader, validation_loader = init(batch_size=args.batch_size, state=args.state, dataset=args.dataset,
                                              input_sizes=input_sizes, mean=mean, std=std, base=base)

        # With best learning rate schedule, warmup proves unnecessary
        # if args.model == 'scnn':
            # Warmup https://github.com/XingangPan/SCNN/issues/82
            # warmup_steps = len(data_loader)
        warmup_steps = 200
        l = lambda t: t / warmup_steps if t < warmup_steps \
            else (1 - (t - warmup_steps) / (len(data_loader) * args.epochs - warmup_steps)) ** 0.9
        # else:
        #    l = lambda t: (1 - t / (len(data_loader) * args.epochs)) ** 0.9
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, l)
        # lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,
        #                                                 lambda t: (1 - t / (len(data_loader) * args.epochs))
        #                                                 ** 0.9)

        # Resume training?
        if args.continue_from is not None:
            load_checkpoint(net=net, optimizer=optimizer, lr_scheduler=lr_scheduler, filename=args.continue_from)

        # Train
        train_schedule(writer=writer, loader=data_loader,
                       validation_loader=None if args.val_num_steps == 0 else validation_loader,
                       criterion=criterion, net=net, optimizer=optimizer, lr_scheduler=lr_scheduler, device=device,
                       num_epochs=args.epochs, is_mixed_precision=args.mixed_precision, input_sizes=input_sizes,
                       exp_name=exp_name, num_classes=num_classes, val_num_steps=args.val_num_steps)

        writer.close()
