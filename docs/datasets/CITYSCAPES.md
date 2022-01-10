# Cityscapes

## Prepare the dataset

1. The dataset can be downloaded in their [official website](https://www.cityscapes-dataset.com/).

2. Change the `CITYSCAPES_ROOT` in [configs/semantic_segmentation/common/datasets/_utils.py](../../configs/semantic_segmentation/common/datasets/_utils.py) to your dataset's location.

3. Pre-processing:

```
python tools/cityscapes_data_list.py
```

## Description

### Directory Structure

```
    ├── <CITYSCAPES.BASE_DIR>                    
        ├── data_lists
        ├── gtFine
        ├── leftImage8bit
        ├── all_demoVideo
        └── ...
```

*More details are coming soon.*
