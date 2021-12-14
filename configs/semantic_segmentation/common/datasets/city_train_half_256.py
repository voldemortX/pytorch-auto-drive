from ._utils import CITY_LABEL_MAP

# Standard cityscapes training with half resolution
# Note: input size is also 1K, while training is conducted at 0.5K, but this seems better for heavy models
train_augmentation = dict(
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
            name='RandomScale',
            min_scale=0.5,
            max_scale=1.5
        ),
        dict(
            name='RandomCrop',
            size=(256, 512)
        ),
        dict(
            name='RandomHorizontalFlip',
            flip_prob=0.5
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
