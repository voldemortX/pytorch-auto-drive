# Aug level: simple
train_augmentation = dict(
    name='Compose',
    transforms=[
        dict(
            name='Resize',
            size_image=(360, 640),
            size_label=(360, 640)
        ),
        dict(
            name='RandomRotation',
            degrees=3
        ),
        dict(
            name='ToTensor'
        ),
        dict(
            name='Normalize',
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            normalize_target=True
        )
    ]
)
