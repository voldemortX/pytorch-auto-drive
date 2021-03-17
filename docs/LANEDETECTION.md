# Lane detection

## Datasets: 

The CULane dataset can be downloaded in their [official website](https://xingangpan.github.io/projects/CULane.html).

The TuSimple dataset can be downloaded at their [github repo](https://github.com/TuSimple/tusimple-benchmark/issues/3). However, you'll also need [segmentation labels](https://drive.google.com/open?id=1LZDCnr79zuNH73NstZ8oIPDud0INCwb9), [list6_train.txt](https://github.com/cardwing/Codes-for-Lane-Detection/blob/master/ENet-TuSimple-Torch/list6/list6_train.txt), [list6_val.txt](https://github.com/cardwing/Codes-for-Lane-Detection/blob/master/ENet-TuSimple-Torch/list6/list6_val.txt) and [list_test.txt](https://github.com/cardwing/Codes-for-Lane-Detection/blob/master/ENet-TuSimple-Torch/list/list_test.txt) provided by [@cardwing](https://github.com/cardwing), thanks for their efforts.

## Training:

1. Change the `BASE_DIR` in [configs.yaml](../configs.yaml) to your datasets' locations.

2. Pre-processing:

For CULane:

```
cp -r <your culane base dir>/list/* <your culane base dir>/lists/
python tools/culane_list_convertor.py
```

For TuSimple:

*First put the data lists you downloaded before in \<your tusimple base dir\>/lists . Then:*

```
python tools/tusimple_list_convertor.py
```

3. Download the ImageNet pre-trained weights *erfnet_encoder_pretrained.pth.tar* from [here](https://github.com/Eromera/erfnet_pytorch/tree/master/trained_models) and put it in the main folder.

4. Training:

```
python main_landec_as_seg.py --epochs=<number of epochs> \
                             --lr=<learning rate> \
                             --batch-size=<any batch size> \ 
                             --dataset=<dataset> \
                             --method=<the method used> \
                             --backbone=<the backbone used> \
                             --exp-name=<whatever you like> \
                             --mixed-precision  # Enable mixed precision

```

We provide directly executable shell scripts for each supported methods in [MODEL_ZOO.md](MODEL_ZOO.md). For detailed instructions, run:

```

python main_landec_as_seg.py --help

```


## Testing:

Training contains online fast validations by using `--val-num-steps=\<some number\>` and the best model is saved, but we find that the best checkpoint is usually the last, so probably no need for validations. For log details you can checkout tensorboard.

To validate a trained model on mean IoU, you can use either mixed-precision or fp32 for any model trained with/without mixed-precision:

```
python main_landec_as_seg.py --state=1 \
                             --continue-from=<path to .pt file> \
                             --dataset=<dataset> \
                             --method=<the method used> \
                             --backbone=<the backbone used> \
                             --batch-size=<any batch size> \
                             --exp-name=<whatever you like> \
                             --mixed-precision  # Enable mixed precision
```

### Test on CULane:

1. Prepare official scripts.

```
cd tools/culane_evaluation
make
mkdir output
chmod 777 eval*
cd -
```

Then change `data_dir` to your CULane base directory in [eval.sh](../tools/culane_evaluation/eval.sh) and [eval_validation.sh](../tools/culane_evaluation/eval_validation.sh). *Mind that you need extra ../../ if relative path is used.*

2. Predict and save lanes.
   
```
python main_landec_as_seg.py --state=<state> \  # 2: test set; 3: validation set           
                             --continue-from=<path to .pt file> \
                             --dataset=<dataset> \ 
                             --method=<the method used> \
                             --backbone=<the backbone used> \ 
                             --batch-size=<any batch size> \  # Recommend 80
                             --mixed-precision  # Enable mixed precision
```

1. Evaluate on the test set with official scripts.

```
./autotest_culane.sh <experiment name, anything is fine> test
```

Or evaluate on the validation set:

```
./autotest_culane.sh <experiment name, anything is fine> val
```

You can then check the test/validation performance at `log.txt`, and per-category performance at `tools/culane_evaluation/output` .

### Test on TuSimple:

1. Prepare official scripts.

```
cd tools/tusimple_evaluation
mkdir output
```

Then change `data_dir` to your TuSimple base directory in [autotest_tusimple.sh](../autotest_tusimple.sh). *Mind that you need extra ../../ if relative path is used.*

2. Predict and save lanes same like CULane.

3. Evaluate on the test set with official scripts.

```
./autotest_tusimple.sh <experiment name, anything is fine> test
```

Or evaluate on the validation set:

```
./autotest_tusimple.sh <experiment name, anything is fine> val
```

You can then check the test/validation performance at `log.txt`, and detailed performance at `tools/tusimple_evaluation/output` .