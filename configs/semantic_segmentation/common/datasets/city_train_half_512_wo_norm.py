from ._utils import CITY_LABEL_MAP

# Standard cityscapes training at half resolution (suitable for small nets: ERFNet/ENet)
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
            name='LabelMap',
            label_id_map=CITY_LABEL_MAP,
            outlier=False
        ),
        dict(
            name='RandomTranslation',
            trans_h=2,
            trans_w=2
        ),
        dict(
            name='RandomHorizontalFlip',
            flip_prob=0.5
        )
        # No normalization for ERFNet
    ]
)
