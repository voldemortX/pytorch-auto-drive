# Semantic segmentation

## Datasets: 

The PASCAL VOC 2012 dataset we use is the commonly used 10582 training set version. If you don't already have that dataset, we refer you to [Google](https://www.google.com) or this [blog](https://www.sun11.me/blog/2018/how-to-use-10582-trainaug-images-on-DeeplabV3-code/).

Other datasets can only be easily downloaded in their official websites.

## Training:

1. Change the `BASE_DIR` in [configs.yaml](../configs.yaml) to your datasets' locations.

2. Pre-processing:

For PASCAL VOC:

*Don't need to do anything.*

For Cityscapes:

```
python tools/cityscapes_data_list.py
```

For GTAV:

```
python tools/gtav_data_list.py
```

For SYNTHIA:

```
python tools/synthia_label_convertor.py
python tools/synthia_data_list.py
```

3. If you are using ERFNet, download the ImageNet pre-trained weights *erfnet_encoder_pretrained.pth.tar* from [here](https://github.com/Eromera/erfnet_pytorch/tree/master/trained_models) and put it in the main folder.

4. Training:

```
python main_landec_as_seg.py --state=<state> \  # 0: normal; 2: decoder training
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
