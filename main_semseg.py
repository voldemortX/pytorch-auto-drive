import os
import time
import torch
import argparse
import math
import yaml
from torch.utils.tensorboard import SummaryWriter
from utils.all_utils_semseg import init, train_schedule, test_one_set, load_checkpoint, build_segmentation_model

if __name__ == '__main__':
    # Settings
    parser = argparse.ArgumentParser(description='PyTorch Auto-drive')
    parser.add_argument('--exp-name', type=str, default='',
                        help='Name of experiment')
    parser.add_argument('--lr', type=float, default=0.002,
                        help='Initial learning rate (default: 0.002)')
    parser.add_argument('--weight-decay', type=float, default=1e-4,
                        help='weight decay (default: 1e-4)')
    parser.add_argument('--epochs', type=int, default=30,
                        help='Number of epochs (default: 30)')
    parser.add_argument('--val-num-steps', type=int, default=1000,
                        help='Validation frequency (default: 1000)')
    parser.add_argument('--workers', type=int, default=8,
                        help='Number of workers (threads) when loading data.'
                             'Recommend value for training: batch_size (default: 8)')
    parser.add_argument('--dataset', type=str, default='voc',
                        help='Train/Evaluate on PASCAL VOC 2012(voc)/Cityscapes(city)/GTAV(gtav)/SYNTHIA(synthia)'
                             '(default: voc)')
    parser.add_argument('--model', type=str, default='deeplabv3',
                        help='Model selection (fcn/erfnet/deeplabv2/deeplabv3/enet/deeplabv2-big/deeplabv3-big)'
                             '(default: deeplabv3)')
    parser.add_argument('--batch-size', type=int, default=8,
                        help='input batch size (default: 8)')
    parser.add_argument('--do-not-save', action='store_false', default=True,
                        help='save model (default: True)')
    parser.add_argument('--mixed-precision', action='store_true', default=False,
                        help='Enable mixed precision training (default: False)')
    parser.add_argument('--continue-from', type=str, default=None,
                        help='Continue training from a previous checkpoint')
    parser.add_argument('--state', type=int, default=0,
                        help='train the whole enet(2)/Conduct final test(1)/normal training(0) (default: 0)')
    parser.add_argument('--encoder-only', action='store_true', default=False,
                        help='Only train the encoder. ENet trains encoder and decoder separately (default: False)')
    args = parser.parse_args()
    exp_name = str(time.time()) if args.exp_name == '' else args.exp_name
    with open(exp_name + '_cfg.txt', 'w') as f:
        f.write(str(vars(args)))
    with open('configs.yaml', 'r') as f:  # Safer and cleaner than box/EasyDict
        configs = yaml.load(f, Loader=yaml.Loader)

    # Basic configurations
    mean = configs['GENERAL']['MEAN']
    std = configs['GENERAL']['STD']
    if args.dataset not in configs['SEGMENTATION_DATASETS'].keys():
        raise ValueError
    num_classes = configs[configs['SEGMENTATION_DATASETS'][args.dataset]]['NUM_CLASSES']
    input_sizes = configs[configs['SEGMENTATION_DATASETS'][args.dataset]]['SIZES']
    categories = configs[configs['SEGMENTATION_DATASETS'][args.dataset]]['CATEGORIES']
    colors = configs[configs['SEGMENTATION_DATASETS'][args.dataset]]['COLORS']
    train_label_id_map = configs[configs['SEGMENTATION_DATASETS'][args.dataset]]['LABEL_ID_MAP'] if \
        'LABEL_ID_MAP' in configs[configs['SEGMENTATION_DATASETS'][args.dataset]].keys() else \
        configs['CITYSCAPES']['LABEL_ID_MAP']
    train_base = configs[configs['SEGMENTATION_DATASETS'][args.dataset]]['BASE_DIR']
    test_base = None
    test_label_id_map = None
    classes = None
    selector = None
    city_aug = 0
    if args.dataset == 'gtav':
        city_aug = 3
        test_base = configs['CITYSCAPES']['BASE_DIR']
        test_label_id_map = configs['CITYSCAPES']['LABEL_ID_MAP']
    elif args.dataset == 'synthia':
        city_aug = 3
        test_base = configs['CITYSCAPES']['BASE_DIR']
        test_label_id_map = configs['CITYSCAPES']['LABEL_ID_MAP']
        classes = 16  # Or 13
        selector = configs['SYNTHIA']['IOU_16']  # Or 13
    device = torch.device('cpu')
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    net, city_aug, input_sizes, weights = build_segmentation_model(configs, args, num_classes, city_aug, input_sizes)
    if weights is not None:
        weights = weights.to(device)
    print(device)
    net.to(device)
    if args.model == 'erfnet' or args.model == 'enet':
        optimizer = torch.optim.Adam(net.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-08,
                                     weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=0.0005)

    # Testing
    if args.state == 1:
        test_loader = init(batch_size=args.batch_size, state=args.state, dataset=args.dataset, input_sizes=input_sizes,
                           mean=mean, std=std, train_base=train_base, test_base=test_base, city_aug=city_aug,
                           train_label_id_map=train_label_id_map, test_label_id_map=test_label_id_map,
                           workers=args.workers)
        load_checkpoint(net=net, optimizer=None, lr_scheduler=None, filename=args.continue_from)
        _, x = test_one_set(loader=test_loader, device=device, net=net, categories=categories, num_classes=num_classes,
                            output_size=input_sizes[2], labels_size=input_sizes[1],
                            is_mixed_precision=args.mixed_precision, selector=selector, classes=classes)
    else:
        criterion = torch.nn.CrossEntropyLoss(ignore_index=255, weight=weights)
        writer = SummaryWriter('runs/' + exp_name)
        train_loader, val_loader = init(batch_size=args.batch_size, state=args.state, dataset=args.dataset,
                                        input_sizes=input_sizes, mean=mean, std=std, train_base=train_base,
                                        test_base=test_base, city_aug=city_aug, workers=args.workers,
                                        train_label_id_map=train_label_id_map, test_label_id_map=test_label_id_map)

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
        if args.continue_from is not None and args.state != 2:
            load_checkpoint(net=net, optimizer=optimizer, lr_scheduler=lr_scheduler, filename=args.continue_from)

        # Train
        train_schedule(writer=writer, loader=train_loader, net=net, optimizer=optimizer, lr_scheduler=lr_scheduler,
                       num_epochs=args.epochs, is_mixed_precision=args.mixed_precision,
                       validation_loader=val_loader, device=device, criterion=criterion, categories=categories,
                       num_classes=num_classes, input_sizes=input_sizes, val_num_steps=args.val_num_steps,
                       classes=classes, selector=selector, encoder_only=args.encoder_only)

        # Final evaluations
        load_checkpoint(net=net, optimizer=None, lr_scheduler=None, filename='temp.pt')
        _, x = test_one_set(loader=val_loader, device=device, net=net, is_mixed_precision=args.mixed_precision,
                            categories=categories, num_classes=num_classes, labels_size=input_sizes[1],
                            output_size=input_sizes[2], encoder_only=args.encoder_only,
                            classes=classes, selector=selector)

        # --do-not-save => args.do_not_save = False
        if args.do_not_save:  # Rename the checkpoint with timestamp
            os.rename('temp.pt', exp_name + '.pt')
        else:  # Since the checkpoint is already saved, it should be deleted
            os.remove('temp.pt')

        writer.close()

    with open('log.txt', 'a') as f:
        f.write(exp_name + ': ' + str(x) + '\n')
