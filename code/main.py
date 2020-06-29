import os
import time
import torch
import argparse
import random
import math
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from apex import amp
from data_processing import colors_voc, colors_city, mean, std, sizes_voc, sizes_city, sizes_city_erfnet, \
                            num_classes_voc, num_classes_city, categories_voc, categories_city, weights_city_erfnet
from deeplab import visualize, init, deeplab_v3, deeplab_v2, fcn, erfnet, train_schedule, test_one_set, load_checkpoint

# All hail Clearlove, 7th of his name!
# torch.manual_seed(4396)
# random.seed(7777)
# np.random.seed(7777)
#torch.backends.cudnn.deterministic = True  # Might hurt performance
#torch.backends.cudnn.benchmark = False  # Might hurt performance


if __name__ == '__main__':
    # Settings
    parser = argparse.ArgumentParser(description='PyTorch 1.2.0')
    parser.add_argument('--lr', type=float, default=0.002,
                        help='Initial learning rate (default: 0.001)')
    parser.add_argument('--epochs', type=int, default=30,
                        help='Number of epochs (default: 30)')
    parser.add_argument('--dataset', type=str, default='voc',
                        help='Train/Evaluate on PASCAL VOC 2012(voc)/Cityscapes(city) (default: voc)')
    parser.add_argument('--model', type=str, default='deeplabv3',
                        help='Model selection (fcn/pspnet/deeplabv2/deeplabv3) (default: deeplabv3)')
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
    if args.dataset == 'voc':
        num_classes = num_classes_voc
        input_sizes = sizes_voc
        categories = categories_voc
        colors = colors_voc
    elif args.dataset == 'city':
        num_classes = num_classes_city
        input_sizes = sizes_city
        categories = categories_city
        colors = colors_city
    else:
        raise ValueError
    device = torch.device('cpu')
    weights = None
    is_erfnet = False
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    if args.model == 'deeplabv3':
        net = deeplab_v3(num_classes=num_classes)
    elif args.model == 'deeplabv2':
        net = deeplab_v2(num_classes=num_classes)
    elif args.model == 'fcn':
        net = fcn(num_classes)
    elif args.model == 'erfnet':
        net = erfnet(num_classes=num_classes)
        weights = torch.tensor(weights_city_erfnet).to(device)
        input_sizes = sizes_city_erfnet
        is_erfnet = True
    else:
        raise ValueError
    print(device)
    net.to(device)

    if args.model == 'erfnet':
        optimizer = torch.optim.Adam(net.parameters(), lr=args.lr, betas=(0.9, 0.999),  eps=1e-08, weight_decay=1e-4)
    else:
        optimizer = torch.optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=0.0005)

    if args.mixed_precision:
        net, optimizer = amp.initialize(net, optimizer, opt_level='O1')

    # Testing
    if args.state == 1:
        test_loader = init(batch_size=args.batch_size, state=args.state, dataset=args.dataset, input_sizes=input_sizes,
                           mean=mean, std=std, erfnet=is_erfnet)
        load_checkpoint(net=net, optimizer=None, lr_scheduler=None,
                        is_mixed_precision=args.mixed_precision, filename=args.continue_from)
        test_one_set(loader=test_loader, device=device, net=net, categories=categories, num_classes=num_classes,
                     output_size=input_sizes[2])
    else:
        criterion = torch.nn.CrossEntropyLoss(ignore_index=255, weight=weights)
        writer = SummaryWriter('runs/experiment_' + str(int(time.time())))
        train_loader, val_loader = init(batch_size=args.batch_size, state=args.state, dataset=args.dataset,
                                        input_sizes=input_sizes, mean=mean, std=std, erfnet=is_erfnet)

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
            load_checkpoint(net=net, optimizer=optimizer, lr_scheduler=lr_scheduler,
                            is_mixed_precision=args.mixed_precision, filename=args.continue_from)
        # visualize(train_loader, colors=colors, mean=mean, std=std)

        # Train
        train_schedule(writer=writer, loader=train_loader, net=net, optimizer=optimizer, lr_scheduler=lr_scheduler,
                       num_epochs=args.epochs, is_mixed_precision=args.mixed_precision, val_num_steps=1000,
                       validation_loader=val_loader, device=device, criterion=criterion, categories=categories,
                       num_classes=num_classes, input_sizes=input_sizes)

        # Final evaluations
        load_checkpoint(net=net, optimizer=None, lr_scheduler=None,
                        is_mixed_precision=args.mixed_precision, filename='temp.pt')
        _, _ = test_one_set(loader=val_loader, device=device, net=net,
                            categories=categories, num_classes=num_classes, output_size=input_sizes[2])

        # --do-not-save => args.do_not_save = False
        if args.do_not_save:  # Rename the checkpoint with timestamp
            os.rename('temp.pt', str(time.time()) + '.pt')
        else:  # Since the checkpoint is already saved, it should be deleted
            os.remove('temp.pt')

        writer.close()
