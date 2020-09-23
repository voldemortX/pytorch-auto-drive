import os
import time
import torch
import argparse
import random
import math
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from losses import LaneLoss, SADLoss
from data_processing import mean, std, sizes_tusimple, sizes_culane, num_classes_tusimple, num_classes_culane
from all_utils_semseg import load_checkpoint
from all_utils_landec import init, train_schedule, test_one_set, erfnet_tusimple, erfnet_culane, \
                             scnn_tusimple, scnn_culane


if __name__ == '__main__':
    # Settings
    parser = argparse.ArgumentParser(description='PyTorch 1.6.0')
    parser.add_argument('--exp-name', type=str, default='',
                        help='Name of experiment')
    parser.add_argument('--lr', type=float, default=0.002,
                        help='Initial learning rate (default: 0.002)')
    parser.add_argument('--epochs', type=int, default=30,
                        help='Number of epochs (default: 30)')
    parser.add_argument('--val-num-steps', type=int, default=1000,
                        help='Validation frequency (default: 1000)')
    parser.add_argument('--dataset', type=str, default='tusimple',
                        help='Train/Evaluate on TuSimple (voc) / CULane (culane) (default: tusimple)')
    parser.add_argument('--model', type=str, default='scnn',
                        help='Model selection (scnn/sad) (default: scnn)')
    parser.add_argument('--batch-size', type=int, default=8,
                        help='input batch size (default: 8)')
    parser.add_argument('--do-not-save', action='store_false', default=True,
                        help='save model (default: True)')
    parser.add_argument('--mixed-precision', action='store_true', default=False,
                        help='Enable mixed precision training (default: False)')
    parser.add_argument('--continue-from', type=str, default=None,
                        help='Continue training from a previous checkpoint')
    parser.add_argument('--state', type=int, default=0,
                        help='Conduct final test(1)/normal training(0) (default: 0)')
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

    exp_name = str(time.time()) if args.exp_name == '' else args.exp_name
    with open(exp_name + '_cfg.txt', 'w') as f:
        f.write(str(vars(args)))

    device = torch.device('cpu')
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    scnn = True if args.model == 'scnn' else False
    if args.dataset == 'tusimple':
        net = erfnet_tusimple(num_classes=num_classes, scnn=scnn)
    elif args.dataset == 'culane':
        net = erfnet_culane(num_classes=num_classes, scnn=scnn)
    else:
        raise ValueError
    print(device)
    net.to(device)

    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr, betas=(0.9, 0.999),  eps=1e-08, weight_decay=1e-4)

    # Testing
    if args.state == 1:
        test_loader = init(batch_size=args.batch_size, state=args.state, dataset=args.dataset, input_sizes=input_sizes,
                           mean=mean, std=std)
        load_checkpoint(net=net, optimizer=None, lr_scheduler=None, filename=args.continue_from)
        test_one_set()
    else:
        # 20200923
        criterion = torch.nn.CrossEntropyLoss()
        writer = SummaryWriter('runs/' + exp_name)
        train_loader, val_loader = init(batch_size=args.batch_size, state=args.state, dataset=args.dataset,
                                        input_sizes=input_sizes, mean=mean, std=std)

        # The "poly" policy, variable names are confusing (May need reimplementation)
        if args.model == 'erfnet':
            # Epoch-wise
            lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,
                                                             lambda x: (1 - math.floor(x / len(train_loader))
                                                                        / args.epochs) ** 0.9)
            # # Original in the paper
            # lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5)
            # # Step lr
            # lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.5)
        else:
            # Step-wise
            lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,
                                                             lambda x: (1 - x / (len(train_loader) * args.epochs))
                                                             ** 0.9)
        # Resume training?
        if args.continue_from is not None:
            load_checkpoint(net=net, optimizer=optimizer, lr_scheduler=lr_scheduler, filename=args.continue_from)
        # visualize(train_loader, colors=colors, mean=mean, std=std)

        # Train
        train_schedule(writer=writer, loader=train_loader, net=net, optimizer=optimizer, lr_scheduler=lr_scheduler,
                       num_epochs=args.epochs, is_mixed_precision=args.mixed_precision,
                       validation_loader=val_loader, device=device, criterion=criterion, categories=categories,
                       num_classes=num_classes, input_sizes=input_sizes, val_num_steps=args.val_num_steps)

        # Final evaluations
        load_checkpoint(net=net, optimizer=None, lr_scheduler=None, filename='temp.pt')
        _, x = test_one_set(loader=val_loader, device=device, net=net, is_mixed_precision=args.mixed_precision,
                            categories=categories, num_classes=num_classes, output_size=input_sizes[2])

        # --do-not-save => args.do_not_save = False
        if args.do_not_save:  # Rename the checkpoint with timestamp
            os.rename('temp.pt', exp_name + '.pt')
        else:  # Since the checkpoint is already saved, it should be deleted
            os.remove('temp.pt')

        writer.close()

        with open('log.txt', 'a') as f:
            f.write(exp_name + ': ' + str(x) + '\n')
