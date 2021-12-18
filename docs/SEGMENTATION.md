# Semantic segmentation

**Before diving into this, please make sure you followed the instructions to prepare datasets in [DATASET.md](./DATASET.md)**

## Training:

If you are using ERFNet, first download the ImageNet pre-trained weights *erfnet_encoder_pretrained.pth.tar* from [here](https://github.com/Eromera/erfnet_pytorch/tree/master/trained_models).

```
python main_semseg.py --train \
                      --config=<config file path> \
                      --mixed-precision  # Enable mixed precision
```

For more instructions that you can override in commandline:

```
python main_semseg.py --help
```
 
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
                      --checkpoint=<ckpt file path> \
                      --mixed-precision  # Enable mixed precision
```

Detail results will be saved to `<save_dir>/<exp_name>/`.

Overall result will be saved to `log.txt`.

Recommend `workers=0 batch_size=1` for high precision inference.
