# Aug level: strong-B (w.o. random lighting, more random affine like LaneATT)
train_augmentation = dict(
    name='Compose',
    transforms=[
        dict(
            name='RandomAffine',
            degrees=(-10, 10),
            scale=(0.8, 1.2),
            translate=(50, 20),
            ignore_x=None
        ),
        dict(
            name='RandomHorizontalFlip',
            flip_prob=0.5,
            ignore_x=None
        ),
        dict(
            name='Resize',
            size_image=(360, 640),
            size_label=(360, 640),
            ignore_x=None
        ),
        dict(
            name='ColorJitter',
            brightness=0.3,
            contrast=0.3,
            saturation=0.3,
            hue=0.15
        ),
        dict(
            name='ToTensor'
        ),
        dict(
            name='Normalize',
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            normalize_target=True,
            ignore_x=None
        )
    ]
)
