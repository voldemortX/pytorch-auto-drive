# Advanced Tutorial

Here we give same advanced use case and coding guide from easy to hard.

## Understand and Write config files

A config file in PytorchAutoDrive (look at [this example](../configs/semantic_segmentation/erfnet/cityscapes_512x1024.py) while reading) is defined by 9 mandatory dicts including:

**Data pipeline:**

1. dataset: the Dataset class
2. train_augmentation: transforms for training
3. test_augmentation: transforms for testing

**Optimization pipeline:**

1. loss: the loss function (e.g. CrossEntropy)
2. optimizer: the optimizer (e.g. SGD)
3. lr_scheduler: the learning rate scheduler (e.g. step lr)

**Model specific options:**

1. train: options for training (e.g. epochs, batch size, DDP options)
2. test: options for evaluation, profiling and visualization (e.g. checkpoint path, image size)
3. model: options to define your model

Also there are 3 optional dicts (beta):

1. vis: used to replace `test` if specified
2. test_dataset: used to replace `dataset` in testing mode if specified
3. vis_dataset: used to replace `dataset` in visualization if specified

Other than `train` and `test`, each dict defines the `__init__` function of a Class, or the input args for a function. Some args are dynamic (e.g. there is a arg for the network instance in optimizers) and will be replaced on the run, which means you don't have to set them in configs. To write a config file is exactly the same as writing Python dicts, you can pitch in maths and imports as well. Do remember to import from `configs.xxx` and most of the first 6 dicts can be directly imported from `configs.xxx.common`. For more details on PytorchAutoDrive's config mechanism, refer to [configs/README.md](../configs/README.md).

## Hyper-parameter tuning and Shortcuts

*Configs are all good and clear, but what if I want to quickly try a set of hyper-parameters for my model? Do I have to write a bunch of config files?*

**The answer is No, you don't have to.**

PytorchAutoDrive provides `--cfg-options` for you to conveniently replace most of the config options in commandline, for instance to replace learning rate to `0.1` and batch size to `32` for ERFNet, you can simply do this:

```
python main_semseg.py --train --config=configs/semantic_segmentation/erfnet/cityscapes_512x1024.py --cfg-options="optimizer.lr=0.1 train.batch_size=32"
```

In practice, you can turn a hyper-parameter search to a shell `search.sh` like this:

```
#!/bin/bash
for lr in 1 2 3; do
  exp_name=erfnet_hyperparameters_${lr}
  python main_semseg.py --train --config=configs/semantic_segmentation/erfnet/cityscapes_512x1024.py --cfg-options="optimizer.lr=0.${lr} train.exp_name=${exp_name}"
  python main_semseg.py --val --config=configs/semantic_segmentation/erfnet/cityscapes_512x1024.py --cfg-options="test.exp_name=${exp_name} test.checkpoint=checkpoints/${exp_name}/model.pt"
done

```

Then you'll see results stored at `checkpoints/` in separate directories named by `exp_name` and tensorboard logs at `checkpoints/tb_logs/`.

The format for `--cfg-options` is `x1=y1 x2=y2`. Argument parsing is based on Python3 `eval()` and supports common `string`, `number`, `list` or `tuple` (even `dict` if you do it right). Don't forget the double quotation marks! 

*--cfg-options is too difficult to write, can I get normal argparse args to use?*

**Yes you can.** You can define **shortcuts** in [statics](../configs/statics.py), we have some preset args in place already, for instance, the equivalent for  `--cfg-options="optimizer.lr=0.1 train.batch_size=32"` is simply: `--lr=0.1 --batch-size=32`. Most of legacy args before the great refactoring are included here.

## The PytorchAutoDrive code trip

PytorchAutoDrive is built by registration, and executed by Runners.

**Algorithms:**

In `utils/` you can find directories containing registered classes or functions for models, datasets, optimizers, etc., and files for utility use. Most algorithm-related codes you need to read can be found here. For instance, you have a dict in config file as:

```
dict(
    name='ERFNet',
    ...
)
```

You can search for the Class/function named `ERFNet` here for its implementation.

**Executions:**

`utils/runners/` implements Runners. A Runner parses config file and constructs the execution process for training/testing/visualization. It defines the behavior of the entire logical process, do read them carefully (best start from `base.py`) if you want to get some deeper customizations from coding.

## Examples

1. [Code a dataset to visualize lane lines and compare with dataset GT](./advanced/VISUALIZE_LANE_DATASETS.md)
2. Code a model: Checkout `utils/models/`, more details is coming...
3. Code a dataset: Checkout `utils/datasets/`, more details is coming...
4. Code a runner for advanced visualization: Checkout `utils/runners/*visualizer.py`, more details is coming...
