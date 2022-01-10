## Configs

Config files in *PytorchAutoDrive* (`./configs/`) are used to define models,
how they are trained, tested, visualized, *etc*.

### Registry Mechanism

Different to existing class-based registers, we can also register functions.
For functions, you only write static args in your config,
while passing the dynamic ones on-the-fly by:

```
REGISTRY.from_dict(
    <config dict for a function/class>,
    kwarg1=1, kwarg2=2, ...
)
```

Note that each argument must be keyword (k=v), and some kwargs can overwrite dict configs.

### Use An Existing Config

Modify customized options like the root of your datasets (in `configs/*/common/_*.py`).

### Write A New Config

Copy the config file most similar to your use case and modify it.
Note that you can simply import config parts from `common` or other config files, it is like writing Python.

### Register A New Class/Func

Choose the appropriate registry and register your Class/Func by:

```
@REGISTRY.register()
```

Remember you still need to import this Class/Func for the registering to take effects.

### How To Read The Code

Since you can't just click 'go to definition' in your IDE,
it is suggested to search the directory for each Class/Function by `name` in configs.
