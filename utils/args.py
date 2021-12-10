import os
from importlib.machinery import SourceFileLoader


def read_config(config_path):
    # Read a mmlab-style python config file and parse to a single dict
    module_name = os.path.split(config_path)[1]
    assert module_name[-3:] == '.py'
    module_name = module_name[:-3]
    module = SourceFileLoader(module_name, config_path).load_module()
    res = {k: module.__dict__[k] for k in module.__all__}
    print(res)

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
        args_dict[k] = v
        cfg[k] = v

    return args, cfg
