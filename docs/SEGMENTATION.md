# Semantic segmentation

**Before diving into this, please make sure you followed the instructions to prepare datasets in [DATASET.md](./DATASET.md)**

**Execution is based on [config files](../configs/README.md)**

## Training:

Some models' ImageNet pre-trained weights need to be manually downloaded, refer to [this table](./IMAGENET_MODELS.md).

```
python main_semseg.py --train \
                      --config=<config file path> \
                      --mixed-precision  # Optional, enable mixed precision \
                      --cfg-options=<overwrite cfg dict>  # Optional
```

Your `<overwrite cfg dict>` is used to manually override config file options in commandline so you don't have to modify config file each time. It should look like this (**the quotation marks are necessary!**): `"train.batch_size=8 train.workers=4 model.classifier_cfg.num_classes=21"`

Some options can be used by shortcuts, such as `--batch-size` will set both `train.batch_size` and `test.batch_size`, for more info:

```
python main_semseg.py --help
```

Example shells are provided in [tools/shells](../tools/shells/).

## Distributed Training

We support multi-GPU training with Distributed Data Parallel (DDP):

```
python -m torch.distributed.launch --nproc_per_node=<number of GPU per-node> --use_env main_semseg.py <your normal args>
```

With DDP, batch size and number of workers are **per-GPU**.

## Testing:

Training contains online evaluations and the best model is saved.

To evaluate a trained model:

```
python main_semseg.py --val \  # No test set labels available
                      --config=<config file path> \
                      --mixed-precision  # Optional, enable mixed precision \
                      --cfg-options=<overwrite cfg dict>  # Optional
```

To test a downloaded pt file, try add `--checkpoint=<pt file path>`.

Detail results will be saved to `<save_dir>/<exp_name>/`.

Overall result will be saved to `log.txt`.

Recommend `workers=0 batch_size=1` for high precision inference.

## Notes:

1. Cityscapes dataset is down-sampled by 2 when training at 256 x 512, to specify different sizes, modify them in config files if needed.

2. All segmentation results reported are from single model without CRF and without multi-scale testing.
