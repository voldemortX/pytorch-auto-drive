# Semantic segmentation

## Datasets: 

The PASCAL VOC 2012 dataset we use is the commonly used 10582 training set version. If you don't already have that dataset, we refer you to [Google](https://www.google.com) or this [blog](https://www.sun11.me/blog/2018/how-to-use-10582-trainaug-images-on-DeeplabV3-code/).

Other datasets can only be downloaded in their official websites.

**Now we assume you are in the code folder.**

## Training:

1. Change the base directories in [code/data_processing.py](code/data_processing.py) to your datasets' locations, variables named `base_*`.

2. Pre-processings:

For PASCAL VOC:

*Don't need to do anything.*

For Cityscapes:

```
python cityscapes_data_list.py
```

For GTAV:

```
python gtav_data_list.py
```

For SYNTHIA:

```
python synthia_label_convertor.py
python synthia_data_list.py
```

3. If you are using ERFNet, download the ImageNet pre-trained weights *erfnet_encoder_pretrained.pth.tar* from [here](https://github.com/Eromera/erfnet_pytorch/tree/master/trained_models), and put it in `code/`.

4. Here are some examples for segmentation:

Mixed precision training on PASCAL VOC 2012 with DeeplabV2:

```
python main_semseg.py --epochs=30 --lr=0.002 --batch-size=8 --dataset=voc --model=deeplabv2 --mixed-precision --exp-name=<whatever you like>
```

Full precision training on Cityscapes with DeeplabV3:

```
python main_semseg.py --epochs=60 --lr=0.002 --batch-size=8 --dataset=city --model=deeplabv3 --exp-name=<whatever you like>
```

Mixed precision training on Cityscapes with ERFNet:

```
python main_semseg.py --epochs=150 --lr=0.0007 --batch-size=10 --dataset=city --model=erfnet --mixed-precision --exp-name=<whatever you like>
```

Mixed precision training on Cityscapes with high resolution DeeplabV2:

```
python main_semseg.py --epochs=60 --lr=0.0014 --batch-size=4 --dataset=city --model=deeplabv2-big --mixed-precision --exp-name=<whatever you like>
```

Mixed precision training on GTAV with high resolution DeeplabV2:

```
python main_semseg.py --epochs=15 --lr=0.0014 --batch-size=4 --dataset=gtav --model=deeplabv2-big --mixed-precision --exp-name=<whatever you like>
```

Mixed precision training on SYNTHIA with high resolution DeeplabV2:

```
python main_semseg.py --epochs=30 --lr=0.0014 --batch-size=4 --dataset=gtav --model=deeplabv2-big --mixed-precision --exp-name=<whatever you like>
```

## Testing:

Training contains online evaluations and the best model is saved, you can check best *val* set performance at `log.txt`, for more details you can checkout tensorboard.

To evaluate a trained model, you can use either mixed-precision or fp32 for any model trained with/without mixed-precision:

```
python main_semseg.py --state=1 --continue-from=<trained model .pt filename> --dataset=<dataset> --model=<trained model architecture> --batch-size=<any batch size>
```
