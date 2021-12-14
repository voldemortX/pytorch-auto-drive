# SYNTHIA standard training
# Note: different label_id_map to others
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
            size=(512, 1024)
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
            label_id_map=[
                255, 10,  2,   0, 1,   4,
                8,   5,   13,  7, 11,  18,
                17,  255, 255, 6, 9,   12,
                14,  15,  16,  3, 255
            ],
            outlier=True
        )
    ]
)
