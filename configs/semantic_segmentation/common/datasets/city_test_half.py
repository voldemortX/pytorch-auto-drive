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
            name='Normalize',
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
        dict(
            name='LabelMap',
            label_id_map=CITY_LABEL_MAP,
            outlier=False
        )
    ]
)
