from torch import optim

from .builder import OPTIMIZERS


@OPTIMIZERS.register()
def torch_optimizer(torch_optim_class, parameters, *args, **kwargs):
    # A direct mapping

    return getattr(optim, torch_optim_class)(parameters, *args, **kwargs)
