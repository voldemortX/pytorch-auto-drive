import torch

from .builder import LR_SCHEDULERS


@LR_SCHEDULERS.register()
def step_scheduler(epochs, len_loader, optimizer, step_ratio=0.9, gamma=0.1):
    return torch.optim.lr_scheduler.StepLR(optimizer, gamma=gamma,
                                           step_size=len_loader * epochs * step_ratio)
