# Visualize & Compare with GT on lane detection datasets

Here we make visualization functions work the same for specific dataset file structures by writing customized Dataset class and the optional `vis_dataset` dict in configs, trough examples.

## Expected Usage

*Note that TuSimple, CULane and LLAMAS are already supported in this manner.*

Write a config file that imports all options from the config file you want to visualize with, and provide a `vis_dataset` dict, e.g., [configs/lane_detection/bezierlanenet/vis_resnet34_tusimple_aug1b.py](../../configs/lane_detection/bezierlanenet/vis_resnet34_tusimple_aug1b.py):

```
from configs.lane_detection.bezierlanenet.resnet34_tusimple_aug1b import *
from configs.lane_detection.common.datasets._utils import TUSIMPLE_ROOT

vis_dataset = dict(
    name='TuSimpleVis',
    root_dataset=TUSIMPLE_ROOT,
    root_output='./test_tusimple_vis',
    keypoint_json="./output/resnet34_bezierlanenet-aug2_tusimple.json",
    image_set='test'
)
```

Then simply run:

```
python tools/vis/lane_img_dir.py --config=configs/lane_detection/bezierlanenet/vis_resnet34_tusimple_aug1b.py --metric tusimple --style line
```

Beware `--style=bezier` can only be used with `--pred`. An example for CULane: [configs/lane_detection/bezierlanenet/vis_resnet34_culane_aug1b.py](../../configs/lane_detection/bezierlanenet/vis_resnet34_culane_aug1b.py)

## Implementation (TuSimple)

The `vis_dataset` can be used to replace `dataset` if specified, as shown in [get_loader()](https://github.com/voldemortX/pytorch-auto-drive/blob/cf314875ff0108f863b9ea8ac8d15141116b8f19/utils/runners/lane_det_visualizer.py#L92) of the default image folder runner. So instead of writing a new runner class, we can simply write a hybrid of [TuSimple](https://github.com/voldemortX/pytorch-auto-drive/blob/cf314875ff0108f863b9ea8ac8d15141116b8f19/utils/datasets/tusimple.py#L15) and [ImageFolderLaneDataset](https://github.com/voldemortX/pytorch-auto-drive/blob/cf314875ff0108f863b9ea8ac8d15141116b8f19/utils/datasets/image_folder.py#L56). It should be able to (functional behavior):
- extract images from the dataset path
- optionally load the dataset GT
- optionally load user-specified predictions

There are two major problems you'll need to address.

1. The training dataset do not load annotations for `val` and `test`, so you'll need to load them same as `train`, and you don't need to append `-2` for training, it would be something like this:

```
def preload_tusimple_labels(json_contents):
    # Load a TuSimple label json's content
    print('Loading json annotation/prediction...')
    targets = []
    for i in tqdm(range(len(json_contents))):
        lines = json_contents[i]['lanes']
        h_samples = json_contents[i]['h_samples']
        temp = []
        for j in range(len(lines)):
            temp.append(np.array([[float(x), float(y)] for x, y in zip(lines[j], h_samples)]))
        targets.append(temp)

    return targets
```

2. The default pipeline uses a `keypoint_process_fn` that is passed dynamically in the runner (not specified in config dict), and processes only the CULane format keypoints in a per-file manner. For TuSimple, where labels are in a single file, it will do best to preload all of them (as above) and replace the `keypoint_process_fn` with a dummy function:

```
def dummy_keypoint_process_fn(label):
    return label

# Then your class `__init__()` will start like this:

class TuSimpleVis(ImageFolderLaneBase):
    def __init__(self, root_dataset, root_output, keypoint_json, image_set, transforms=None,
                 keypoint_process_fn=None, use_gt=True):
        super().__init__(root_dataset, root_output, transforms, dummy_keypoint_process_fn)
```

With help of existing base classes, the full code is provided as reference in [utils/datasets/tusimple_vis.py](../../utils/datasets/tusimple_vis.py).

