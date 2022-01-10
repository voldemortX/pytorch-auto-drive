# Standard scale crop flip, scale is important
train_augmentation = dict(
    name='Compose',
    transforms=[
        dict(
            name='ToTensor'
        ),
        dict(
            name='RandomScale',
            min_scale=0.5,
            max_scale=1.5
        ),
        dict(
            name='RandomCrop',
            size=(321, 321)
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
