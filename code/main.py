from torchsummary import summary
import time
import torch
from torch.utils.tensorboard import SummaryWriter
import argparse
from deeplab import visualize, init, deeplab_v3, train_schedule, test_one_set


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
    parser.add_argument('--continue-from', type=str, default=None,
                        help='Continue training from a previous checkpoint')
    parser.add_argument('--state', type=int, default=0,
                        help='Conduct final test(2)/final training(1)/normal training(0) (default: 0)')
    args = parser.parse_args()

    # Basic configurations
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    net = deeplab_v3()
    print(device)
    net.to(device)

    # Testing
    if args.state == 2:
        test_loader, categories = init(batch_size=args.batch_size, state=args.state)
        net.load_state_dict(torch.load(args.continue_from))
        test_one_set(loader=test_loader, device=device, net=net)

    else:
        #summary(net, (3, 513, 513))  # Seems not working with a loaded model
        criterion = torch.nn.CrossEntropyLoss(ignore_index=255)
        writer = SummaryWriter('runs/experiment_' + str(int(time.time())))
        # Resume training?
        if args.continue_from is not None:
            net.load_state_dict(torch.load(args.continue_from))

        # Final training
        if args.state == 1:
            train_loader, categories = init(batch_size=args.batch_size, state=args.state)
            visualize(train_loader, categories)

            # Train
            train_schedule(writer=writer, loader=train_loader, initial_lr=args.lr, num_epochs=args.epochs, net=net,
                           with_validation=False, validation_loader=None, device=device, criterion=criterion)

            # Final evaluations
            train_acc = test_one_set(loader=train_loader, device=device, net=net)

        # Normal training
        elif args.state == 0:
            train_loader, val_loader, categories = init(batch_size=args.batch_size, state=args.state)
            visualize(train_loader, categories)

            # Train
            train_schedule(writer=writer, loader=train_loader, initial_lr=args.lr, num_epochs=args.epochs, net=net,
                           with_validation=True, validation_loader=val_loader, device=device, criterion=criterion)

            # Final evaluations
            train_acc = test_one_set(loader=train_loader, device=device, net=net)
            val_acc = test_one_set(loader=val_loader, device=device, net=net)

        # Save parameters(#epoch is not saved)
        if args.do_not_save:  # --do-not-save => args.do_not_save = False
            torch.save(net.state_dict(), str(time.time()) + '.pth')

        writer.close()
