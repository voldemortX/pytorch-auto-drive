import torch
from collections import OrderedDict

from .ddp_utils import save_on_master


def get_warnings():
    # Get rid of the extra line of code printing
    # https://stackoverflow.com/a/26433913/15449902
    import warnings

    def warning_on_one_line(message, category, filename, lineno, file=None, line=None):
        return '%s:%s: %s: %s\n' % (filename, lineno, category.__name__, message)

    warnings.formatwarning = warning_on_one_line

    return warnings


warnings = get_warnings()


# Save model checkpoints (supports amp)
def save_checkpoint(net, optimizer, lr_scheduler, filename='temp.pt'):
    checkpoint = {
        'model': net.state_dict(),
        'optimizer': optimizer.state_dict() if optimizer is not None else None,
        'lr_scheduler': lr_scheduler.state_dict() if lr_scheduler is not None else None
    }
    save_on_master(checkpoint, filename)


# Load model checkpoints (supports amp)
def load_checkpoint(net, optimizer, lr_scheduler, filename):
    try:
        checkpoint = torch.load(filename, map_location='cpu')
    except:
        warnings.warn('Model not saved as on cpu, could be a legacy trained weight, trying loading on saved device...')
        checkpoint = torch.load(filename)
        print('Loaded on saved device.')

    # To keep BC while having a acceptable variable name for lane detection
    checkpoint['model'] = OrderedDict((k.replace('aux_head', 'lane_classifier') if 'aux_head' in k else k, v)
                                      for k, v in checkpoint['model'].items())
    net.load_state_dict(checkpoint['model'])

    if optimizer is not None:
        try:  # Shouldn't be necessary, but just in case
            optimizer.load_state_dict(checkpoint['optimizer'])
        except RuntimeError:
            warnings.warn('Incorrect optimizer state dict, maybe you are using old code with aux_head?')
            pass
    if lr_scheduler is not None:
        try:  # Shouldn't be necessary, but just in case
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        except RuntimeError:
            warnings.warn('Incorrect lr scheduler state dict, maybe you are using old code with aux_head?')
            pass
