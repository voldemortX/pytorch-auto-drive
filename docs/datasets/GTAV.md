# GTAV

## Prepare the dataset

1. The dataset can be downloaded in their [official website](https://download.visinf.tu-darmstadt.de/data/from_games/).

2. Change the `GTAV_ROOT` in [configs/semantic_segmentation/common/datasets/_utils.py](../../configs/semantic_segmentation/common/datasets/_utils.py) to your dataset's location.

3. Pre-processing:

```
python tools/gtav_data_list.py
```

## Description

### Directory Structure

```
    ├── <GTAV.BASE_DIR>                    
        ├── data_lists
        ├── images
        └── labels
```

*More details are coming soon.*
