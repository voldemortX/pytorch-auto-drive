# Lane detection

**Before diving into this, please make sure you followed the instructions to prepare datasets in [DATASET.md](./DATASET.md)**

**Execution is based on [config files](../configs/README.md)**

## Training:

Some models' ImageNet pre-trained weights need to be manually downloaded, refer to [this table](./IMAGENET_MODELS.md).

```
python main_landet.py --train \
                      --config=<config file path> \
                      --mixed-precision  # Optional, enable mixed precision \
                      --cfg-options=<overwrite cfg dict>  # Optional
```

Your `<overwrite cfg dict>` is used to manually override config file options in commandline so you don't have to modify config file each time. It should look like this (**the quotation marks are necessary!**): `"train.batch_size=8 train.workers=4 model.lane_classifier_cfg.dropout=0.1"`

Some options can be used by shortcuts, such as `--batch-size` will set both `train.batch_size` and `test.batch_size`, for more info:

```
python main_landet.py --help
```

Example shells are provided in [tools/shells](../tools/shells/).

## Distributed Training

We support multi-GPU training with Distributed Data Parallel (DDP):

```
python -m torch.distributed.launch --nproc_per_node=<number of GPU per-node> --use_env main_landet.py <your normal args>
```

With DDP, batch size and number of workers are **per-GPU**.

## Testing

### Evaluation:

1. Predict lane lines:

```
python main_landet.py --test \  # Or --val for validation
                      --config=<config file path> \
                      --mixed-precision  # Optional, enable mixed precision \
                      --cfg-options=<overwrite cfg dict>  # Optional
```

To test a downloaded pt file, try add `--checkpoint=<pt file path>`.

Note that LLAMAS doesn't have test set labels.

2. Test with official scripts on `<my_dataset>`:

```
./autotest_<my_dataset>.sh <exp_name> <mode> <save_dir>
```

`<mode>` includes `test` and `val`.

`<save_dir>` and `<exp_name>` are recommended to set the same as in config file, so detail evaluation results will be saved to `<save_dir>/<exp_name>/`

Overall result will be saved to `log.txt`.

### Fast evaluation in mIoU [Not Recommended]:

Training contains online fast validations by using `val_num_steps` and the best model is saved, but we find that the best checkpoint is usually the last, so probably no need for validations. For log details you can checkout tensorboard.

To validate a trained model on mean IoU, you can use either mixed-precision or fp32 for any model trained with/without mixed-precision:

```
python main_landet.py --valfast \
                      --config=<config file path> \
                      --mixed-precision  # Optional, enable mixed precision \
                      --cfg-options=<overwrite cfg dict>  # Optional
```
