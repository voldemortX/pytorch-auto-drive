import os
import warnings
from importlib.machinery import SourceFileLoader

ACTION_ARGS = [
    'mixed_precision'
]  # Args that can't be set in configs


def read_config(config_path):
    # Read a mmlab-style python config file and parse to a single dict
    module_name = os.path.split(config_path)[1]
    assert module_name[-3:] == '.py'
    module_name = module_name[:-3]
    module = SourceFileLoader(module_name, config_path).load_module()
    res = {k: v for k, v in module.__dict__.items() if not k.startswith('__')}

    return res


def parse_arg_cfg(args, cfg, defaults=None):
    # args > config > defaults
    args_dict = vars(args)  # Linked changes
    for k in args_dict.keys():
        if k == 'config':
            continue
        if args_dict[k] is not None:  # 1
            v = args_dict[k]
        else:
            if k in cfg.keys():  # 2
                v = cfg[k]
            else:  # 3
                v = defaults[k]

        if k in ACTION_ARGS and k in cfg.keys() and cfg[k] != v:
            warnings.warn('Bool arg `{}={}` in config is illegal, replaced by commandline setting `{}={}`.'.format(
                k, cfg[k], k, v
            ))
        args_dict[k] = v
        cfg[k] = v

    return args, cfg
