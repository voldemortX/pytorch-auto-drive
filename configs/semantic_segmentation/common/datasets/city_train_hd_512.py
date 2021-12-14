from ._utils import CITY_LABEL_MAP

# Standard cityscapes crop training
train_augmentation = dict(
    name='Compose',
    transforms=[
        dict(
            name='ToTensor'
        ),
        dict(
            name='RandomCrop',
            size=(512, 1024)
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
        ),
        dict(
            name='Normalize',
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ]
)
