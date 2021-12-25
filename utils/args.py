import os
from importlib.machinery import SourceFileLoader
try:
    import ujson as json
except ImportError:
    import json

from configs.statics import DEPRECATION_MAP, SHORTCUTS
try:
    from .common import warnings
except ImportError:
    import warnings


def update_nested(d, keys, value):
    # Update nested dict with keys as list
    if not isinstance(d, dict):
        raise ValueError
    if len(keys) == 1:
        d[keys[0]] = value
    else:
        if keys[0] in d.keys():
            d[keys[0]] = update_nested(d[keys[0]], keys[1:], value)
        else:
            d[keys[0]] = update_nested({}, keys[1:], value)

    return d


def cmd_dict(x):
    # A data type to hack a dict-style argparse input,
    # every key should be in k1.k2.kn format to the last non-dict value,
    # values can't include tuples,
    # other more complex settings should refer to config files instead.
    # x: x1=y1 x2=y2 x3=y3 etc.
    options = x.split()
    res = {}
    for o in options:
        kv = o.split('=', maxsplit=1)
        res[kv[0]] = json.loads(kv[1])

    return res


def add_shortcuts(parser):
    # TODO: duplicates
    for k, v in SHORTCUTS.items():
        parser.add_argument('--' + k.replace('_', '-'), type=v['type'],
                            help='{}. Shortcut for {}'.format(v['help'], str(v['keys'])))


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


def parse_arg_cfg(args, cfg, deprecation_map=None):
    if deprecation_map is None:
        deprecation_map = DEPRECATION_MAP

    # Simply reject deprecations
    dict_args = vars(args)
    if deprecation_map is not None:
        for deprecated, v in deprecation_map.items():
            if dict_args.get(deprecated) is not None:
                if v['valid'] is None:  # Not used anymore
                    warnings.warn('Deprecated arg {}={} will not be used. '.format(deprecated, cfg[deprecated])
                                  + v['message'])
                else:  # Use the deprecated arg in absence of new arg
                    warnings.warn('Arg {} is deprecated, please use {} instead. '.format(deprecated, v['valid'])
                                  + v['message'])
                    if dict_args.get(v['valid']) is None:
                        dict_args[v['valid']] = dict_args[deprecated]

    # Set shortcuts
    overrides = dict_args['cfg_options']
    if overrides is None:
        overrides = {}
    for k, v in dict_args.items():
        if k in SHORTCUTS.keys():
            for tk in SHORTCUTS[k]['keys']:
                v_cfg_options = overrides.get(tk)
                if v_cfg_options is not None:
                    if v is not None and v != v_cfg_options:
                        raise ValueError('Conflict between arg {}={} in --cfg-option and shortcut arg {}={}'.format(
                            tk, v_cfg_options, k, v
                        ))
                else:
                    overrides[tk] = v

    # Override cfg by args
    for k, v in overrides.items():
        if v is not None:
            if type(v) == bool:
                warnings.warn('Override Bool arg {} is insecure, by default, it will be overridden by False!'.format(
                    k
                ))
            # k = 'k1.k2.kn'
            key_path = k.split('.')
            try:
                cfg = update_nested(cfg, key_path, v)
            except RuntimeError:
                raise RuntimeError('Structural conflict in config key path {}!'.format(key_path))

    # Add retain args

    return args, cfg
