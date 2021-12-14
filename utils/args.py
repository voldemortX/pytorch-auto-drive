import os
from importlib.machinery import SourceFileLoader
try:
    from .common import warnings
except ImportError:
    import warnings


def read_config(config_path):
    # Read a mmlab-style python config file and parse to a single dict
    module_name = os.path.split(config_path)[1]
    assert module_name[-3:] == '.py'
    module_name = module_name[:-3]
    module = SourceFileLoader(module_name, config_path).load_module()
    res = {k: v for k, v in module.__dict__.items() if not k.startswith('__')}

    return res


def map_states(args, states):
    # Map --train --test etc. to args.state
    state = args.state
    for k, v in vars(args).items():
        if k in states and v:  # Map states
            state = states.index(k)

    return state


def parse_arg_cfg(args, cfg, defaults=None, required=None, deprecation_map=None):
    # args > config > defaults
    # required: [...]
    # deprecation_map: {deprecated_arg: {valid: new_arg, message: custom warning message}, ...}
    args_dict = vars(args)  # Linked changes
    for k in args_dict.keys():
        if required is not None and k in required:  # Required args are set only at commandline
            v = args_dict[k]
            if k in cfg.keys():
                warnings.warn('Required arg `{}={}` in config is illegal, replaced by commandline\'s `{}={}`.'.format(
                    k, cfg[v], k, v
                ))
        elif args_dict[k] is not None:  # 1
            v = args_dict[k]
        else:
            if k in cfg.keys():  # 2
                v = cfg[k]
            else:  # 3
                v = defaults[k]

        if type(args_dict[k]) == bool and k in cfg.keys() and cfg[k] != v:
            warnings.warn('Bool arg `{}={}` in config is illegal, replaced by commandline\'s `{}={}`.'.format(
                k, cfg[k], k, v
            ))
        args_dict[k] = v
        cfg[k] = v

    # Handle simple deprecations
    if deprecation_map is not None:
        for deprecated, v in deprecation_map.items():
            if cfg[deprecated] is None or ('expected' in v.keys() and cfg[deprecated] == v['expected']):
                continue
            if v['valid'] is None:
                warnings.warn('Deprecated arg {}={} will not be used. '.format(deprecated, cfg[deprecated])
                              + v['message'])
            elif type(cfg[deprecated]) == bool or type(cfg[v['valid']]) == bool or cfg[v['valid']] is not None:
                warnings.warn('Deprecated arg {}={} will be overridden with new arg {}={}. '.format(
                    deprecated, cfg[deprecated], v['valid'], cfg[v['valid']]
                ) + v['message'])
            else:  # Use the deprecated arg in absence of new arg
                warnings.warn('Arg {} is deprecated, please use {} instead. '.format(deprecated, v['valid'])
                              + v['message'])
                cfg[v['valid']] = cfg[deprecated]
                args_dict[v['valid']] = args_dict[deprecated]

    return args, cfg
