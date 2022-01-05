from torch import optim

from .builder import OPTIMIZERS


def set_weight_decay(model, no_decay_keywords=()):
    has_decay = []
    no_decay = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights
        if len(param.shape) == 1 or name.endswith(".bias") or check_keywords_in_name(name, no_decay_keywords):
            no_decay.append(param)
            # print(f"{name} has no weight decay")
        else:
            has_decay.append(param)
    return [{'params': has_decay},
            {'params': no_decay, 'weight_decay': 0.}]


def check_keywords_in_name(name, keywords=()):
    isin = False
    for keyword in keywords:
        if keyword in name:
            isin = True
    return isin


@OPTIMIZERS.register()
def torch_optimizer(torch_optim_class, model, filter_group=None, *args, **kwargs):
    # A direct mapping
    if filter_group is not None:
        parameters = set_weight_decay(model)
    else:
        parameters = model.parameters()

    return getattr(optim, torch_optim_class)(parameters, *args, **kwargs)
