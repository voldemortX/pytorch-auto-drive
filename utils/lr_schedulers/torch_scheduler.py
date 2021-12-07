from torch.optim import lr_scheduler

from .builder import LR_SCHEDULERS


@LR_SCHEDULERS.register()
def torch_scheduler(torch_optim_class, optimizer, *args, **kwargs):
    # A direct mapping

    return getattr(lr_scheduler, torch_optim_class)(optimizer, *args, **kwargs)
