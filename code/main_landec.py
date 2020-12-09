import time
import torch
import argparse
from torch.utils.tensorboard import SummaryWriter
from losses import LaneLoss, SADLoss
from data_processing import mean, std, sizes_tusimple, sizes_culane, num_classes_tusimple, num_classes_culane, \
                            weights_tusimple, weights_culane, gap_tusimple, gap_culane, ppl_culane, ppl_tusimple
from all_utils_semseg import load_checkpoint
from all_utils_landec import init, train_schedule, test_one_set, erfnet_tusimple, erfnet_culane, fast_evaluate


if __name__ == '__main__':
    # Settings
    parser = argparse.ArgumentParser(description='PyTorch 1.6.0')
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
    parser.add_argument('--model', type=str, default='scnn',
                        help='Model selection (baseline/scnn/sad) (default: scnn)')
    parser.add_argument('--batch-size', type=int, default=8,
                        help='input batch size (default: 8)')
    parser.add_argument('--mixed-precision', action='store_true', default=False,
                        help='Enable mixed precision training (default: False)')
    parser.add_argument('--continue-from', type=str, default=None,
                        help='Continue training from a previous checkpoint')
    parser.add_argument('--state', type=int, default=0,
                        help='Conduct final test(2)/validate(1)/normal training(0) (default: 0)')
    args = parser.parse_args()

    # Basic configurations
    if args.dataset == 'tusimple':
        num_classes = num_classes_tusimple
        input_sizes = sizes_tusimple
    elif args.dataset == 'culane':
        num_classes = num_classes_culane
        input_sizes = sizes_culane
    else:
        raise ValueError

    states = ['train', 'val', 'test']
    exp_name = str(time.time()) if args.exp_name == '' else args.exp_name
    with open(exp_name + '_' + states[args.state] + '_cfg.txt', 'w') as f:
        f.write(str(vars(args)))

    device = torch.device('cpu')
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    scnn = True if args.model == 'scnn' else False
    weights = None
    if args.dataset == 'tusimple':
        weights = torch.tensor(weights_tusimple).to(device)
        net = erfnet_tusimple(num_classes=num_classes, scnn=scnn)
        gap = gap_tusimple
        ppl = ppl_tusimple
    elif args.dataset == 'culane':
        weights = torch.tensor(weights_culane).to(device)
        net = erfnet_culane(num_classes=num_classes, scnn=scnn)
        gap = gap_culane
        ppl = ppl_culane
    else:
        raise ValueError
    print(device)
    net.to(device)

    optimizer = torch.optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)
    # optimizer = torch.optim.Adam(net.parameters(), lr=args.lr, betas=(0.9, 0.999),  eps=1e-08, weight_decay=1e-4)

    # Testing
    if args.state == 1 or args.state == 2:
        data_loader = init(batch_size=args.batch_size, state=args.state, dataset=args.dataset, input_sizes=input_sizes,
                           mean=mean, std=std)
        load_checkpoint(net=net, optimizer=None, lr_scheduler=None, filename=args.continue_from)
        if args.state == 1:  # Validate with mean IoU
            _, x = fast_evaluate(loader=data_loader, device=device, net=net,
                                 num_classes=num_classes, output_size=input_sizes[0],
                                 is_mixed_precision=args.mixed_precision)
            with open('log.txt', 'a') as f:
                f.write(exp_name + ' validation: ' + str(x) + '\n')

        else:  # Test with official scripts later (so just predict lanes here)
            test_one_set(net=net, device=device, loader=data_loader, is_mixed_precision=args.mixed_precision,
                         input_sizes=input_sizes, gap=gap, ppl=ppl, dataset=args.dataset)
    else:
        if args.model == 'scnn' or args.model == 'baseline':
            criterion = LaneLoss(weight=weights, ignore_index=255)
        elif args.model == 'sad':
            criterion = SADLoss()
        else:
            raise NotImplementedError

        writer = SummaryWriter('runs/' + exp_name)
        data_loader, validation_loader = init(batch_size=args.batch_size, state=args.state, dataset=args.dataset,
                                              input_sizes=input_sizes, mean=mean, std=std)

        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,
                                                         lambda x: (1 - x / (len(data_loader) * args.epochs))
                                                         ** 0.9)
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
