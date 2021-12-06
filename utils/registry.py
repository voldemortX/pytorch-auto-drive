class SimpleRegistry(object):
    # A simple portable registry that can register and init/run class/function, inspired by
    # mmcv's Registry
    # fvcore's Registry
    # Implementing this ourselves to retain the possibility of not using above complex packages.

    def __init__(self) -> None:
        self._map = {}

    def register(self):
        # Suppose to work as @name.register()
        def decorator(function_or_class):
            name = function_or_class.__name__
            # Register instead of execute
            if name in self._map.keys():
                raise ValueError('Conflicting name for registered Function or Class {}'.format(name))
            self._map[name] = function_or_class
            return function_or_class

        return decorator

    def get(self, name):
        res = self._map.get(name)
        if res is None:
            raise KeyError('Class or Function {} not found in registry {}!'.format(name, self._name))

        return res

    def from_dict(self, dict_params, **kwargs):
        dict_params_ = dict_params.copy()
        dict_params_.update(kwargs)
        name = dict_params_.pop('name')
        function_or_class = self.get(name)

        return function_or_class(**dict_params_)
