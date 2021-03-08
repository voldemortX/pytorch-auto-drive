# Welcome to pytorch-auto-drive visualization tutorial

Use `--help` option for detailed commandline instructions. Colors can be specified in [configs.yaml](../configs.yaml) for each dataset.

[vis_tools.py](../tools/vis_tools.py) contains batch-wise visualization functions to modify for your own use case.

## Segmentation mask

Use [visualize_segmentation.py](../visualize_segmentation.py) to visualize segmentation results, by providing the image with `--image-path` and mask (**not the colored ones**) with `--mask-path`, also `--dataset` needs to be specified for color selection.

For example, visualize on PASCAL VOC 2012:

```
python visualize_segmentation.py --image-path=test_images/voc_test_image.jpg --mask-path=test_images/voc_test_mask.png --save-path=test_images/voc_test.png --dataset=voc
```

You should be able to see the result like this stored at `--save-path`:

<div align="center">
  <img src="vis_voc1.png"/>
</div>

If mask is not provided, an inference will be performed by the model specified with `--model` and `--continue-from`, you can define input resolution with `--height` and `--width`, but the result will always be resized to the original image:

```
python visualize_segmentation.py --image-path=test_images/voc_test_image.jpg --save-path=test_images/voc_pred.png --model=deeplabv2 --dataset=voc --mixed-precision --continue-from=<deeplabv2_pascalvoc_321x321_20201108.pt> --height=505 --width=505
```

## Lane points

Use [visualize_lane.py](../visualize_lane.py) to visualize lane detection results.

By providing a mask with `--mask-path`, lanes will be drawn as non-transparent segmentation masks:

```
python visualize_lane.py --image-path=test_images/culane_test_image.jpg --mask-path=test_images/culane_test_mask.png --save-path=test_images/culane_test.png --dataset=culane
```

The result will be like this:

<div align="center">
  <img src="vis_culane2.png"/>
</div>

You can also draw sample points with `--keypoint-path` in the CULane format, for example:

```
python visualize_lane.py --image-path=test_images/culane_test_image.jpg --keypoint-path=test_images/culane_test_keypoint.txt --save-path=test_images/culane_test.png --dataset=culane
```

<div align="center">
  <img src="vis_culane1.png"/>
</div>

Sample points and segmentation mask can be drawn together if both files are provided.
