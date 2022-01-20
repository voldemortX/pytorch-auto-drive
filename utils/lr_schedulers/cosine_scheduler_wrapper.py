from torch.optim import lr_scheduler

from .builder import LR_SCHEDULERS


@LR_SCHEDULERS.register()
def CosineAnnealingLRWrapper(epochs, len_loader, optimizer):
    # Wrap it so that len_loader is not required in configs
    return lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=len_loader * epochs)