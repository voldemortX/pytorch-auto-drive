from torch import nn

from .builder import LOSSES


@LOSSES.register()
def torch_loss(torch_loss_class, *args, **kwargs):
    # A direct mapping

    return getattr(nn, torch_loss_class)(*args, **kwargs)
