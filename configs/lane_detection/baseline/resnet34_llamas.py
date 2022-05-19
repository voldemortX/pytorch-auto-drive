from importmagician import import_from
with import_from('./'):
    # Data pipeline
    from configs.lane_detection.common.datasets.llamas_seg import dataset
    from configs.lane_detection.common.datasets.train_level0_360 import train_augmentation
    from configs.lane_detection.common.datasets.test_360 import test_augmentation

    # Optimization pipeline
    from configs.lane_detection.common.optims.segloss_5class import loss
    from configs.lane_detection.common.optims.sgd05 import optimizer
    from configs.lane_detection.common.optims.ep18_poly_warmup200 import lr_scheduler


train = dict(
    exp_name='resnet34_baseline_llamas',
    workers=10,
    batch_size=20,
    checkpoint=None,
    # Device args
    world_size=0,
    dist_url='env://',
    device='cuda',

    val_num_steps=0,  # Seg IoU validation (mostly useless)
    save_dir='./checkpoints',

    input_size=(360, 640),
    original_size=(717, 1276),
    num_classes=5,
    num_epochs=18,
    collate_fn=None,  # 'dict_collate_fn' for LSTR
    seg=True  # Seg-based method or not
)

test = dict(
    exp_name='resnet34_baseline_llamas',
    workers=10,
    batch_size=80,
    checkpoint='./checkpoints/resnet34_baseline_llamas/model.pt',
    # Device args
    device='cuda',

    save_dir='./checkpoints',

    seg=True,
    gap=1,
    ppl=417,
    thresh=0.3,
    collate_fn=None,  # 'dict_collate_fn' for LSTR
    input_size=(360, 640),
    original_size=(717, 1276),
    max_lane=4,
    dataset_name='llamas'
)

# Essentially DeepLabV1 without dilation like in SCNN paper
model = dict(
    name='standard_segmentation_model',
    backbone_cfg=dict(
        name='predefined_resnet_backbone',
        backbone_name='resnet34',
        return_layer='layer4',
        pretrained=True,
        replace_stride_with_dilation=[False, True, True]
    ),
    reducer_cfg=dict(
        name='RESAReducer',
        in_channels=512,
        reduce=128
    ),
    classifier_cfg=dict(
        name='DeepLabV1Head',
        in_channels=128,
        num_classes=5,
        dilation=1
    ),
    lane_classifier_cfg=dict(
        name='SimpleLaneExist',
        num_output=5 - 1,
        flattened_size=4400
    )
)
