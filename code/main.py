from torchsummary import summary
import os
import time
import torch
import argparse
import random
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from apex import amp
from deeplab import visualize, init, deeplab_v3, train_schedule, test_one_set, load_checkpoint

# All hail Clearlove, 7th of his name!
torch.manual_seed(4396)
random.seed(7777)
np.random.seed(7777)
torch.backends.cudnn.deterministic = True  # Might hurt performance
torch.backends.cudnn.benchmark = False  # Might hurt performance


def after_loading():
    global lr_scheduler
    # The "poly" policy, variable names are confusing(May need reimplementation)
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,
                                                     lambda x: (1 - x / (len(train_loader) * args.epochs)) ** 0.9)
    # Resume training?
    if args.continue_from is not None:
        load_checkpoint(net=net, optimizer=optimizer, lr_scheduler=lr_scheduler,
                        is_mixed_precision=args.mixed_precision, filename=args.continue_from)
    visualize(train_loader)


if __name__ == '__main__':
    # Settings
    parser = argparse.ArgumentParser(description='PyTorch 1.3')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Initial learning rate (default: 0.001)')
    parser.add_argument('--epochs', type=int, default=30,
                        help='Number of epochs (default: 30)')
    parser.add_argument('--batch-size', type=int, default=4,
                        help='input batch size (default: 4)')
    parser.add_argument('--do-not-save', action='store_false', default=True,
                        help='save model (default: True)')
    parser.add_argument('--mixed-precision', action='store_true', default=False,
                        help='Enable mixed precision training (default: False)')
    parser.add_argument('--continue-from', type=str, default=None,
                        help='Continue training from a previous checkpoint')
    parser.add_argument('--state', type=int, default=0,
                        help='Conduct final test(2)/final training(1)/normal training(0) (default: 0)')
    parser.add_argument('--gpu-id', type=str, default='0',
                        help='Specify which GPU is to be used (default: 0)')
    args = parser.parse_args()

    # Basic configurations
    device = torch.device('cpu')
    if torch.cuda.is_available():
        device = torch.device('cuda:' + args.gpu_id)
    net = deeplab_v3()
    print(device)
    net.to(device)
    optimizer = torch.optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=0.0001)
    if args.mixed_precision:
        net, optimizer = amp.initialize(net, optimizer, opt_level='O1')

    # Testing
    if args.state == 2:
        test_loader = init(batch_size=args.batch_size, state=args.state)
        load_checkpoint(net=net, optimizer=None, lr_scheduler=None,
                        is_mixed_precision=args.mixed_precision, filename=args.continue_from)
        test_one_set(loader=test_loader, device=device, net=net)
    else:
        #summary(net, (3, 513, 513))  # Seems not working with a loaded model
        criterion = torch.nn.CrossEntropyLoss(ignore_index=255)
        writer = SummaryWriter('runs/experiment_' + str(int(time.time())))

        # Final training
        if args.state == 1:
            train_loader = init(batch_size=args.batch_size, state=args.state)
            after_loading()

            # Train
            train_schedule(writer=writer, loader=train_loader, net=net, optimizer=optimizer, lr_scheduler=lr_scheduler,
                           num_epochs=args.epochs, is_mixed_precision=args.mixed_precision,
                           with_validation=False, validation_loader=None, device=device, criterion=criterion)

            # Final evaluations
            train_acc = test_one_set(loader=train_loader, device=device, net=net)

        # Normal training
        elif args.state == 0:
            train_loader, val_loader = init(batch_size=args.batch_size, state=args.state)
            after_loading()

            # Train
            train_schedule(writer=writer, loader=train_loader, net=net, optimizer=optimizer, lr_scheduler=lr_scheduler,
                           num_epochs=args.epochs, is_mixed_precision=args.mixed_precision,
                           with_validation=True, validation_loader=val_loader, device=device, criterion=criterion)

            # Final evaluations(not necessary)
            train_acc = test_one_set(loader=train_loader, device=device, net=net)
            val_acc = test_one_set(loader=val_loader, device=device, net=net)

        # --do-not-save => args.do_not_save = False
        if args.do_not_save:  # Rename the checkpoint with timestamp
            os.rename('temp.pt', str(time.time()) + '.pt')
        else:  # Since the checkpoint is already saved, it should be deleted
            os.remove('temp.pt')

        writer.close()
