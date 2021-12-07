from torch.optim import optimizer

from .builder import OPTIMIZERS


@OPTIMIZERS.register()
def torch_optimizer(torch_optim_class, parameters, *args, **kwargs):
    # A direct mapping

    return getattr(optimizer, torch_optim_class)(parameters, *args, **kwargs)
