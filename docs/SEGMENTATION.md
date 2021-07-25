# Semantic segmentation

## Training:

If you are using ERFNet, first download the ImageNet pre-trained weights *erfnet_encoder_pretrained.pth.tar* from [here](https://github.com/Eromera/erfnet_pytorch/tree/master/trained_models) and put it in the main folder.

```
python main_semseg.py --state=<state> \  # 0: normal; 2: decoder training
                      --epochs=<number of epochs> \
                      --lr=<learning rate> \
                      --batch-size=<any batch size> \ 
                      --dataset=<dataset> \
                      --model=<the model used> \
                      --exp-name=<whatever you like> \
                      --mixed-precision \  # Enable mixed precision
                      --encoder-only  # Pre-train encoder
```

We provide directly executable shell scripts for each supported models in [MODEL_ZOO.md](MODEL_ZOO.md). You can run a shell script (e.g. `xxx.sh`) by:

```
./tools/shells/xxx.sh
```

For detailed instructions, run:

```
python main_semseg.py --help
```

## Distributed Training

We support multi-GPU training with Distributed Data Parallel (DDP):

```
python -m torch.distributed.launch --nproc_per_node=<number of GPU per-node> --use_env main_semseg.py --world-size=<total GPU> --dist-url=<socket url like tcp://localhost:23456> <your normal args>
```

With DDP, `--batch-size` means batch size per-GPU, and more dataloader threads should be used with `--workers`.

## Testing:

Training contains online evaluations and the best model is saved, you can check best *val* set performance at `log.txt`, for more details you can checkout tensorboard.

To evaluate a trained model, you can use either mixed-precision or fp32 for any model trained with/without mixed-precision:

```
python main_semseg.py --state=1 \
                      --continue-from=<trained model .pt filename> \
                      --dataset=<dataset> \
                      --model=<the model used> \ 
                      --batch-size=<any batch size> \
                      --mixed-precision  # Enable mixed precision
```

Recommend `--workers=0 --batch-size=1` for high precision inference.
