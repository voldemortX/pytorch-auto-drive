# Aug level: strong
train_augmentation = dict(
    name='Compose',
    transforms=[
        dict(
            name='RandomApply',
            apply_prob=10/11,
            transforms=[
                dict(
                    name='RandomRotation',
                    degrees=10
                ),
                dict(
                    name='RandomHorizontalFlip',
                    flip_prob=0.5
                ),
                dict(
                    name='RandomCrop',
                    size=(int(720 * 0.9), int(1280 * 0.9))
                )
            ]
        ),
        dict(
            name='Resize',
            size_image=(360, 640),
            size_label=(360, 640)
        ),
        dict(
            name='ColorJitter',
            brightness=0.4,
            contrast=0.4,
            saturation=0.4,
            hue=0.2
        ),
        dict(
            name='ToTensor'
        ),
        dict(
            name='RandomLighting',
            mean=0.0,
            std=0.1,
            eigen_value=[0.00341571, 0.01817699, 0.2141788],
            eigen_vector=[
                [0.41340352, -0.69563484, -0.58752847],
                [-0.81221408, 0.00994535, -0.5832747],
                [0.41158938, 0.71832671, -0.56089297]
            ]
        ),
        dict(
            name='Normalize',
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            normalize_target=True
        )
    ]
)
