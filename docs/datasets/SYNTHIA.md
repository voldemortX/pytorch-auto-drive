# SYNTHIA

## Prepare the dataset

1. The dataset can be downloaded in their [official website](http://synthia-dataset.net/downloads/).

2. Change the `SYNTHIA.BASE_DIR` in [configs.yaml](../../configs.yaml) to your dataset's location.

3. Pre-processing:

```
python tools/synthia_label_convertor.py
python tools/synthia_data_list.py
```

## Description

### Directory Structure

```
    ├── <SYNTHIA.BASE_DIR>                 
        ├── data_lists
        ├── RGB
        ├── GT
        └── ...
```

*More details are coming soon.*
