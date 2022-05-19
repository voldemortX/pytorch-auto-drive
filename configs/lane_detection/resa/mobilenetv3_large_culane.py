from importmagician import import_from
with import_from('./'):
    # Data pipeline
    from configs.lane_detection.common.datasets.culane_seg import dataset
    from configs.lane_detection.common.datasets.train_level0_288 import train_augmentation
    from configs.lane_detection.common.datasets.test_288 import test_augmentation

    # Optimization pipeline
    from configs.lane_detection.common.optims.segloss_5class import loss
    from configs.lane_detection.common.optims.sgd006 import optimizer
    from configs.lane_detection.common.optims.ep12_poly_warmup500 import lr_scheduler


train = dict(
    exp_name='mobilenetv3-large_resa_culane',
    workers=10,
    batch_size=20,
    checkpoint=None,
    # Device args
    world_size=0,
    dist_url='env://',
    device='cuda',

    val_num_steps=0,  # Seg IoU validation (mostly useless)
    save_dir='./checkpoints',

    input_size=(288, 800),
    original_size=(590, 1640),
    num_classes=5,
    num_epochs=12,
    collate_fn=None,  # 'dict_collate_fn' for LSTR
    seg=True  # Seg-based method or not
)

test = dict(
    exp_name='mobilenetv3-large_resa_culane',
    workers=10,
    batch_size=80,
    checkpoint='./checkpoints/mobilenetv3-large_resa_culane/model.pt',
    # Device args
    device='cuda',

    save_dir='./checkpoints',

    seg=True,
    gap=20,
    ppl=18,
    thresh=0.3,
    collate_fn=None,  # 'dict_collate_fn' for LSTR
    input_size=(288, 800),
    original_size=(590, 1640),
    max_lane=4,
    dataset_name='culane'
)

model = dict(
    name='RESA_Net',
    backbone_cfg=dict(
        # MobileNetV3-Large 1.0
        # Manually download https://download.pytorch.org/models/mobilenet_v3_large-8738ca79.pth,
        # bug in torch 1.6
        name='MobileNetV3Encoder',
        pretrained='mobilenet_v3_large-8738ca79.pth',
        arch='large',
        reduction_factor=1,

        # OS-16 (DeepLab style)
        strides=(1, 2, 1, 2, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1),
        dilations=(1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2),
        out_indices=(16, )
    ),
    spatial_conv_cfg=dict(
        name='RESA',
        num_channels=128,
        iteration=4,
        alpha=2.0
    ),
    classifier_cfg=dict(
        name='BUSD',
        in_channels=128,
        num_classes=5
    ),
    reducer_cfg=dict(
        name='RESAReducer',
        in_channels=960,
        reduce=128
    ),
    lane_classifier_cfg=dict(
        name='EDLaneExist',
        num_output=5 - 1,
        flattened_size=1125,
        dropout=0.1,
        pool='avg'
    )
)
