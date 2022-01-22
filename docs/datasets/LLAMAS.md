# LLAMAS

## Prepare the dataset

1. The LLAMAS dataset can be downloaded in their [official website](https://unsupervised-llamas.com/llamas/).

2. Change the `LLAMAS_ROOT` in [configs/lane_detection/common/datasets/_utils.py](../../configs/lane_detection/common/datasets/_utils.py) to your dataset's location..

3. Pre-processing:

```
  python tools/llamas_list_convertor.py
```

LLAMAS dataset provides both color and gray images. We use color images in our framework.

4. Prepare official evaluation scripts:

```
cd tools/llamas_evaluation
mkdir output
```

Then change `data_dir` to your LLAMAS base directory in [autotest_llamas.sh](../../autotest_llamas.sh). *Mind that you need extra ../../ if relative path is used.*

5. If you use Bézier curve methods, download Bézier curve GT from [here](https://drive.google.com/file/d/1klugsr5lT9SxW0r5oyYjfo2qplt1TyxU/view?usp=sharing) and unzip them in `LLAMAS_ROOT/bezier_labels/`. More info on curves are in [CURVE.md](../CURVE.md).

## Description

### Directory Structure

```
  <LLAMAS.BASE_DIR>
     ├─ color_images    
     │  ├─ test
     │  ├─ train
     │  └─ valid
     ├─ labels    
     │  ├─ train
     │  └─ valid
     ├─ laneseg_labels
     │  ├─ train
     │  └─ valid
     ├─ bezier_labels
     │  ├─ train_3.json
     │  └─ ... 
     └─ lists
```

The test set' s annotations are not public.

### Label Data Format

```
{
    "image_name": "...",
    "projection_matrix": [[x11, 0, x13], [0, x21, x23], [0, 0, 1]],
    "lanes":
        [
            {
                "lane_id": "...", 
                "markers": 
                 [
                     {
                        "lane_marker_id": "...", 
                        "world_start":{"x":"...", "y": "...", "z": "..."},  "pixel_start": {"x": "...", "y": "..."},
                        "world_end":{"x":"...", "y": "...", "z": "..."}, "pixel_start": {"x": "...", "y": "..."}}
                     },
                     ...                 
                 ]
            },
            ...     
        ]
}
```

LLAMAS dataset employs json files to save annotations. Each image corresponds to a json file. 

We utilize [the format of culane](CULANE.md) to reformat LLAMAS dataset. 

We use the [script](https://github.com/XingangPan/seg_label_generate) provided by Xingang Pan to generate per-pixel labels.

The generated labels can be downloaded from [Google Drive](https://drive.google.com/file/d/1XA4nRLuAzsjJUSUs4HCjz7dksI9dHDNd/view?usp=sharing).





