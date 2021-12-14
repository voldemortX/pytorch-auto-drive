from ._utils import CITY_LABEL_MAP

# Standard cityscapes testing at 2K
# Note: input size is also 2K, while training is conducted at 1K, but this seems better for heavy models
test_augmentation = dict(
    name='Compose',
    transforms=[
        dict(
            name='ToTensor'
        ),
        dict(
            name='Resize',
            size_image=(1024, 2048),
            size_label=(1024, 2048)
        ),
        dict(
            name='Normalize',
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
        dict(
            name='LabelMap',
            label_id_map=CITY_LABEL_MAP
        )
    ]
)
