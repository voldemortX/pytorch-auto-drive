from ._utils import CITY_LABEL_MAP

# Standard cityscapes testing at half resolution
test_augmentation = dict(
    name='Compose',
    transforms=[
        dict(
            name='ToTensor'
        ),
        dict(
            name='Resize',
            size_image=(512, 1024),
            size_label=(512, 1024)
        ),
        dict(
            name='LabelMap',
            label_id_map=CITY_LABEL_MAP
        )
        # No normalization for ERFNet
    ]
)
