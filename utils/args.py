import os
from importlib.machinery import SourceFileLoader


def read_config(config_path):
    # Read a mmlab style python config file and parse to a single dict
    module_name = os.path.split(config_path)[1]
    module = SourceFileLoader(module_name, config_path)
    print(module.__dict__)
    return module.__dict__


def parse_arg_cfg(args, cfg, defaults=None):
    args_dict = vars(args)  # Linked changes
    for k, _ in args_dict.keys():
        if k == 'config':
            continue
        if args_dict[k] is not None:  # Set in args, use args
            v = args_dict[k]
        else:  # Both unset, use defaults
            if k not in cfg.keys():
                v = defaults[k]
            else:  # Not set in args, set in cfg, use cfg
                v = cfg[k]
        args_dict[k] = v
        cfg[k] = v

    return args, cfg
