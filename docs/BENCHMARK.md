# Welcome to pytorch-auto-drive benchmark

## Lane detection performance

| method | backbone | resolution | FPS | FLOPS(G) | Params(M) |
| :---: | :---: | :---: | :---: | :---: | :---: |
| Baseline | VGG16 | 360 x 640 | 56.36 | 214.50 | 20.37 | 
| Baseline | ResNet18 | 360 x 640 | 148.59 | 85.24 | 12.04 | 
| Baseline | ResNet34 | 360 x 640 | 79.97 | 159.60 | 22.15 |
| Baseline | ResNet50 | 360 x 640 | 50.58 | 177.62 | 24.57 |
| Baseline | ResNet101 | 360 x 640 | 27.41 | 314.36 | 43.56 |
| Baseline | ERFNet | 360 x 640 | 85.87 | 26.32 | 2.66 | 
| Baseline | ENet | 360 x 640 | 56.63 | 4.26 | 0.95 |  
| SCNN | VGG16 | 360 x 640 | 21.18 | 218.64 | 20.96 |
| SCNN | ResNet18 | 360 x 640 | 21.12 | 89.38 | 12.63 | 
| SCNN | ResNet34 | 360 x 640 | 20.77 | 163.74 | 22.74 | 
| SCNN | ResNet50 | 360 x 640 | 19.59 | 181.76 | 25.16 |
| SCNN | ResNet101 | 360 x 640 | 13.50 | 318.50 | 44.15 | 
| SCNN | ERFNet | 360 x 640 | 18.40 | 30.46 | 3.25 | 
| Baseline | VGG16 | 288 x 800 | 55.31 | 214.50 | 20.15 | 
| Baseline | ResNet18 | 288 x 800 | 136.28 | 85.22 | 11.82 | 
| Baseline | ResNet34 | 288 x 800 | 72.42 | 159.60 | 21.93 | 
| Baseline | ResNet50 | 288 x 800 | 49.41 | 177.60 | 24.35 | 
| Baseline | ResNet101 | 288 x 800 | 27.19 | 314.34 | 43.34 | 
| Baseline | ERFNet | 288 x 800 | 88.76 | 26.26 | 2.68 | 
| Baseline | ENet | 288 x 800 | 57.99 | 4.12 | 0.96 | 
| SCNN | VGG16 | 288 x 800 | 21.40 | 218.62 | 20.74 | 
| SCNN | ResNet18 | 288 x 800 | 20.80 | 89.34 | 12.42 | 
| SCNN | ResNet34 | 288 x 800 | 19.77 | 163.72 | 22.52 | 
| SCNN | ResNet50 | 288 x 800 | 18.88 | 181.72 | 24.94 | 
| SCNN | ResNet101 | 288 x 800 | 13.42 | 318.46 | 43.94 | 
| SCNN | ERFNet | 288 x 800 | 18.80 | 30.40 | 3.27 | 

## Segmentation performance:

| method | resolution  | FPS | FLOPS(G) | Params(M) |
| :---: | :---: | :---: | :---: | :---: |
| FCN | 256 x 512 | 43.32 | 216.42 | 51.95 |
| FCN | 512 x 1024 | 12.06 | 865.69 | 51.95 |
| FCN | 1024 x 2048 | 3.06 | 3462.77 | 51.95 |
| ERFNet | 256 x 512 | 91.20 | 15.03 | 2.06 |
| ERFNet | 512 x 1024 | 85.51 | 60.11 | 2.06 |
| ERFNet | 1024 x 2048 | 21.53 | 240.44 | 2.06 |
| ENet | 256 x 512 | 59.31 | 2.72 | 0.35 |
| ENet | 512 x 1024 | 55.69 | 10.88 | 0.35 |
| ENet | 1024 x 2048 | 30.88 | 43.53 | 0.35 |
| DeeplabV2 | 256 x 512 | 44.87 | 180.59 | 43.20 |
| DeeplabV2 | 512 x 1024 | 12.93 | 722.37 | 43.20 |
| DeeplabV2 | 1024 x 2048 | 3.23 | 2889.49 | 43.20 |
| DeeplabV3 | 256 x 512 | 35.26 | 241.65 | 58.63 |
| DeeplabV3 | 512 x 1024 | 10.26 | 966.61 | 58.63 |
| DeeplabV3 | 1024 x 2048 | 2.56 | 3866.45| 58.63 |

*All results are the maximum value of 3 times on a RTX 2080Ti.*

## Test examples

In the setting of `mode=simple`, we employ a random tensor to replace the real image. 
Based on this operation, we can avoid using the DataLoader so as to obtain the best fps of models.

For lane detection:

```
python profiling.py  --task=lane \           
                      --times=3 \
                      --dataset=<dataset name> \
                      --method=<the method used> \
                      --backbone=<the backbone used> \
                      --mode=simple \
                      --height=<the height of choosing dataset> \
                      --width=<the width of choosing dataset>
```

For segmentation:

```
python profiling.py  --task=seg \           
                      --times=3 \
                      --dataset=<dataset name> \
                      --model=<the model used> \
                      --mode=simple \
                      --height=<the height of choosing dataset> \
                      --width=<the width of choosing dataset>
```

In the setting of `mode=real`, In order to simulate that the real camera transmit frames to models, we set 'batch_size=1' and 'num_workers=0' in the DataLoader.

For lane detection:

```
python profiling.py   --task=lane \           
                      --times=3 \
                      --dataset=<dataset name> \
                      --method=<the method used> \
                      --backbone=<the backbone used> \
                      --mode=real \
                      --height=<the height of choosing dataset> \
                      --width=<the width of choosing dataset> \
                      --continue-from=<pre-trained model>
```

For segmentation:

```
python profiling.py   --task=seg \           
                      --times=3 \
                      --dataset=<dataset name> \
                      --model=<the model used> \
                      --mode=real \
                      --height=<the height of choosing dataset> \
                      --width=<the width of choosing dataset> \
                      --continue-from=<pre-trained model>
```