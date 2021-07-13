# Provide a dummy torch.amp utils for better coverage of lower PyTorch versions
import functools


class autocast(object):
    def __init__(self, enabled=True):
        pass

    def __enter__(self):
        pass

    def __exit__(self, *args):
        pass

    def __call__(self, func):
        @functools.wraps(func)
        def decorate_autocast(*args, **kwargs):
            with self:
                return func(*args, **kwargs)
        return decorate_autocast


class GradScaler(object):
    def __init__(self,
                 init_scale=2.**16,
                 growth_factor=2.0,
                 backoff_factor=0.5,
                 growth_interval=2000,
                 enabled=True):
        pass

    def scale(self, outputs):
        return outputs

    def step(self, optimizer, *args, **kwargs):
        return optimizer.step(*args, **kwargs)

    def update(self):
        pass
