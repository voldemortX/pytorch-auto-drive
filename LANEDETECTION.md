# Lane detection

## Datasets: 

The CULane dataset can be downloaded in their [official website](https://xingangpan.github.io/projects/CULane.html).

The TuSimple dataset can be downloaded at their [github repo](https://github.com/TuSimple/tusimple-benchmark/issues/3). However, you'll also need [segmentation labels](https://drive.google.com/open?id=1LZDCnr79zuNH73NstZ8oIPDud0INCwb9), [list6_train.txt](https://github.com/cardwing/Codes-for-Lane-Detection/blob/master/ENet-TuSimple-Torch/list6/list6_train.txt), [list6_val.txt](https://github.com/cardwing/Codes-for-Lane-Detection/blob/master/ENet-TuSimple-Torch/list6/list6_val.txt) and [list_test.txt](https://github.com/cardwing/Codes-for-Lane-Detection/blob/master/ENet-TuSimple-Torch/list/list_test.txt) provided by [@cardwing](https://github.com/cardwing), thanks for their efforts.

## Training:

1. Change the `BASE_DIR` in [configs.yaml](configs.yaml) to your datasets' locations.

2. Pre-processing:

For CULane:

```
cp -r <your culane base dir>/list/* <your culane base dir>/lists/
python tools/culane_list_convertor.py
```

For TuSimple:

*First put the data lists you downloaded before in \<your tusimple base dir\>/lists .*

```
python tools/tusimple_list_convertor.py
```

3. Download the ImageNet pre-trained weights *erfnet_encoder_pretrained.pth.tar* from [here](https://github.com/Eromera/erfnet_pytorch/tree/master/trained_models) and put it in the main folder.

4. Here are some examples for lane detection:

Mixed precision training on CULane with ERFNet:

```
python main_landec_as_seg.py --epochs=12 --lr=0.2 --batch-size=20 --dataset=culane --method=baseline --backbone=erfnet --mixed-precision --exp-name=<whatever you like>
```

Mixed precision training on TuSimple with ERFNet:

```
python main_landec_as_seg.py --epochs=50 --lr=0.2 --batch-size=20 --dataset=tusimple --method=baseline --backbone=erfnet --mixed-precision --exp-name=<whatever you like>
```

Mixed precision training on CULane with ERFNet-SCNN:

```
python main_landec_as_seg.py --epochs=12 --lr=0.2 --batch-size=20 --dataset=culane --method=scnn --backbone=erfnet --mixed-precision --exp-name=<whatever you like>
```

Mixed precision training on TuSimple with ERFNet-SCNN:

```
python main_landec_as_seg.py --epochs=50 --lr=0.2 --batch-size=20 --dataset=tusimple --method=scnn --backbone=erfnet --mixed-precision --exp-name=<whatever you like>
```

Mixed precision training on CULane with VGG-SCNN:

```
python main_landec_as_seg.py --epochs=12 --lr=0.015 --batch-size=20 --dataset=culane --method=scnn --backbone=vgg16 --mixed-precision --exp-name=<whatever you like>
```


## Testing:

Training contains online fast validations by using --val-num-steps=\<some number > 0\> and the best model is saved, but we find that the best checkpoint is usually the last, so probably no need for validations. For log details you can checkout tensorboard.

To validate a trained model on mean IoU, you can use either mixed-precision or fp32 for any model trained with/without mixed-precision:

```
python main_landec_as_seg.py --state=1 --continue-from=<trained model .pt filename> --dataset=<dataset> --method=<trained method architecture> --backbone=<trained backbone> --batch-size=<any batch size> --exp-name=<whatever you like> --mixed-precision
```

### Test a trained model with CULane:

1. Prepare official scripts.

```
cd tools/culane_evaluation
make
mkdir output
chmod 777 eval*
cd -
```

Then change `data_dir` to your CULane base directory in [eval.sh](tools/culane_evaluation/eval.sh) and [eval_validation.sh](tools/culane_evaluation/eval_validation.sh). *Mind that you need extra ../../ if relative path is used.*

2. Predict and save lanes.
   
```
python main_landec_as_seg.py --state=2 --continue-from=<trained model .pt filename> --dataset=<dataset> --method=<trained model architecture> --backbone=<trained backbone> --batch-size=<any batch size, recommend 80> --mixed-precision
```

Use `--state=3` to predict lanes for the validation set.

3. Evaluate on the test set with official scripts.

```
./autotest_culane.sh <your experiment name> test
```

Or evaluate on the validation set:

```
./autotest_culane.sh <your experiment name> val
```

You can then check the test/validation performance at `log.txt`, and per-class performance at `tools/culane_evaluation/output` .

### Test a trained model with TuSimple:

1. Prepare official scripts.

```
cd tools/tusimple_evaluation
mkdir output
```

Then change `data_dir` to your TuSimple base directory in [autotest_tusimple.sh](autotest_tusimple.sh). *Mind that you need extra ../../ if relative path is used.*

2. Predict and save lanes.
   
```
python main_landec_as_seg.py --state=2 --continue-from=<trained model .pt filename> --dataset=<dataset> --method=<trained model architecture> --backbone=<trained backbone> --batch-size=<any batch size, recommend 80> --mixed-precision
```

Use `--state=3` to predict lanes for the validation set.

3. Evaluate on the test set with official scripts.

```
./autotest_tusimple.sh <your experiment name> test
```

Or evaluate on the validation set:

```
./autotest_tusimple.sh <your experiment name> val
```

You can then check the test/validation performance at `log.txt`, and detailed performance at `tools/tusimple_evaluation/output` .
