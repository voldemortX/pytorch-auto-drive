# GTAV

## Prepare the dataset

1. The dataset can be downloaded in their [official website](https://download.visinf.tu-darmstadt.de/data/from_games/).

2. Change the `GTAV.BASE_DIR` in [configs.yaml](../../configs.yaml) to your dataset's location.

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
