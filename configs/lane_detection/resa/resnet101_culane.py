from importmagician import import_from
with import_from('./'):
    # Data pipeline
    from configs.lane_detection.common.datasets.culane_seg import dataset
    from configs.lane_detection.common.datasets.train_level0_288 import train_augmentation
    from configs.lane_detection.common.datasets.test_288 import test_augmentation

    # Optimization pipeline
    from configs.lane_detection.common.optims.segloss_5class import loss
    from configs.lane_detection.common.optims.sgd0048 import optimizer
    from configs.lane_detection.common.optims.ep12_poly_warmup600 import lr_scheduler


train = dict(
    exp_name='resnet101_resa_culane',
    workers=2,
    batch_size=2,
    checkpoint=None,
    # Device args
    world_size=8,
    dist_url='tcp://localhost:12345',
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
    exp_name='resnet101_resa_culane',
    workers=2,
    batch_size=8,
    checkpoint='./checkpoints/resnet101_resa_culane/model.pt',
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
        name='predefined_resnet_backbone',
        backbone_name='resnet101',
        return_layer='layer3',
        pretrained=True,
        replace_stride_with_dilation=[False, True, True]
    ),
    reducer_cfg=dict(
        name='RESAReducer',
        in_channels=1024,
        reduce=128
    ),
    spatial_conv_cfg=dict(
        name='RESA',
        num_channels=128,
        iteration=5,
        alpha=2.0
    ),
    classifier_cfg=dict(
        name='BUSD',
        in_channels=128,
        num_classes=5
    ),
    lane_classifier_cfg=dict(
        name='EDLaneExist',
        num_output=5 - 1,
        flattened_size=4500,
        dropout=0.1,
        pool='avg'
    )
)
