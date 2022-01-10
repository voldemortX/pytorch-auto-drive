import math
from torch.optim import lr_scheduler

from .builder import LR_SCHEDULERS


@LR_SCHEDULERS.register()
def epoch_poly_scheduler(epochs, len_loader, optimizer, power=0.9):
    # Poly scheduler for ERFNet
    return lr_scheduler.LambdaLR(
        optimizer,
        lambda x: (1 - math.floor(x / len_loader) / epochs) ** power)


@LR_SCHEDULERS.register()
def poly_scheduler(epochs, len_loader, optimizer, power=0.9):
    # Poly scheduler
    return lr_scheduler.LambdaLR(
        optimizer,
        lambda x: (1 - x / (len_loader * epochs)) ** power)


@LR_SCHEDULERS.register()
def poly_scheduler_with_warmup(epochs, len_loader, optimizer, power=0.9, warmup_steps=0):
    # Poly scheduler with warmup
    def f(t):  # PEP8-E731
        return t / warmup_steps if t < warmup_steps \
            else (1 - (t - warmup_steps) / (len_loader * epochs - warmup_steps)) ** power
    return lr_scheduler.LambdaLR(optimizer, f)
