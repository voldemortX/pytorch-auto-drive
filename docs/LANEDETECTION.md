# Lane detection

**Before diving into this, please make sure you followed the instructions to prepare datasets in [DATASET.md](./DATASET.md)**

**Execution is based on [config files](../configs/README.md)**

## Training:

If you are using ERFNet, first download the ImageNet pre-trained weights *erfnet_encoder_pretrained.pth.tar* from [here](https://github.com/Eromera/erfnet_pytorch/tree/master/trained_models).

```
python main_landec.py --train \
                      --config=<config file path> \
                      --mixed-precision  # Enable mixed precision

```

For more instructions that you can override in commandline:

```
python main_landec.py --help
```

## Distributed Training

We support multi-GPU training with Distributed Data Parallel (DDP):

```
python -m torch.distributed.launch --nproc_per_node=<number of GPU per-node> --use_env main_landec.py <your normal args>
```

With DDP, batch size and number of workers are **per-GPU**.

## Testing

### Evaluation:

1. Predict lane lines:

```
python main_landec.py --test \  # Or --val for validation
                      --config=<config file path> \
                      --checkpoint=<ckpt file path> \
                      --mixed-precision  # Enable mixed precision
```

Note that LLAMAS doesn't have test set labels.

2. Test with official scripts on `<my_dataset>`:

```
./autotest_<my_dataset>.sh <exp_name> <mode> <save_dir>
```

`<mode>` includes `test` and `val`.

`<save_dir>` and `<exp_name>` are recommended to set the same as in config file, so detail evaluation results will be saved to `<save_dir>/<exp_name>/`.

Overall result will be saved to `log.txt`.

### Fast evaluation in mIoU [Not Recommended]:

Training contains online fast validations by using `val_num_steps` and the best model is saved, but we find that the best checkpoint is usually the last, so probably no need for validations. For log details you can checkout tensorboard.

To validate a trained model on mean IoU, you can use either mixed-precision or fp32 for any model trained with/without mixed-precision:

```
python main_landec.py --valfast \
                      --config=<config file path> \
                      --checkpoint=<ckpt file path> \
                      --mixed-precision  # Enable mixed precision
```
