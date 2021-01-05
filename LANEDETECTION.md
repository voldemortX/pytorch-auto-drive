# Lane detection

## Datasets: 

The CULane dataset can be downloaded in their [official website](https://xingangpan.github.io/projects/CULane.html).

The TuSimple dataset can be downloaded at their [github repo](https://github.com/TuSimple/tusimple-benchmark/issues/3). However, you'll also need [segmentation labels](https://drive.google.com/open?id=1uLZk_i6rxRMvwLF8dLy19dTJiOgnbotf), [train data lists](https://drive.google.com/open?id=1hzfxufoCnUYEahQ3k29b8flJhNk0gAo4) and [test data list](https://github.com/cardwing/Codes-for-Lane-Detection/blob/master/ENet-TuSimple-Torch/list/list_test.txt) provided by [@cardwing](https://github.com/cardwing), thanks for their efforts.

**Now we assume you are in the code folder.**

## Training:

1. Change the base directories in [code/data_processing.py](code/data_processing.py) to your datasets' locations, variables named `base_*`.

2. Pre-processing:

For CULane:

```
cp -r <your culane base dir>/list <your culane base dir>/lists
python culane_list_convertor.py
```

For TuSimple:

*First put the data lists you downloaded before in \<your tusimple base dir\>/lists .*

```
python tusimple_list_convertor.py
```

3. Download the ImageNet pre-trained weights *erfnet_encoder_pretrained.pth.tar* from [here](https://github.com/Eromera/erfnet_pytorch/tree/master/trained_models), and put it in `code/`.

4. Here are some examples for lane detection:

Mixed precision training on CULane with ERFNet:

```
python main_landec.py --epochs=12 --lr=0.15 --batch-size=20 --dataset=culane --model=erfnet --mixed-precision --exp-name=<whatever you like>
```

Mixed precision training on CULane with ERFNet-SCNN:

```
python main_landec.py --epochs=12 --lr=0.08 --batch-size=20 --dataset=culane --model=scnn --mixed-precision --exp-name=<whatever you like>
```

## Testing:

Training contains online fast validations by using --val-num-steps=\<some number > 0\> and the best model is saved, but we find that the best checkpoint is usually the last, so probably no need for validations. For log details you can checkout tensorboard.

To validate a trained model on mean IoU, you can use either mixed-precision or fp32 for any model trained with/without mixed-precision:

```
python main_landec.py --state=1 --continue-from=<trained model .pt filename> --dataset=<dataset> --model=<trained model architecture> --batch-size=<any batch size> --exp-name=<whatever you like> --mixed-precision
```

### To test a trained model with CULane:

1. Prepare official scripts.

```
cd tools/culane_evaluation
make
mkdir output
```

*Then change `data_dir` to your CULane base directory in [eval.sh](code/tools/culane_evaluation/eval.sh).*

2. Predict and save lanes.
   
```
python main_landec.py --state=2 --continue-from=<trained model .pt filename> --dataset=<dataset> --model=<trained model architecture> --batch-size=<any batch size, recommend 80> --mixed-precision
```

3. Evaluate with official scripts.

```
./autotest_culane.sh <your experiment name>
```

You can then check the test/validation performance at `log.txt`, and per-class performance at `code/tools/culane_evaluation/output` .

### To test a trained model with TuSimple:

1. Prepare official scripts.

```
cd tools/tusimple_evaluation
mkdir output
```

*Then change `data_dir` to your TuSimple base directory in [autotest_tusimple.sh](code/autotest_tusimple.sh).*

1. Predict and save lanes.
   
```
python main_landec.py --state=2 --continue-from=<trained model .pt filename> --dataset=<dataset> --model=<trained model architecture> --batch-size=<any batch size, recommend 80> --mixed-precision
```

3. Evaluate with official scripts.

```
./autotest_tusimple.sh <your experiment name>
```

You can then check the test/validation performance at `log.txt`, and detailed performance at `code/tools/tusimple_evaluation/output` .
