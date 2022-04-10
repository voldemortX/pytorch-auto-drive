# Data pipeline
from configs.lane_detection.common.datasets.tusimple_seg import dataset
from configs.lane_detection.common.datasets.train_level0_360 import train_augmentation
from configs.lane_detection.common.datasets.test_360 import test_augmentation

# Optimization pipeline
from configs.lane_detection.common.optims.segloss_7class import loss
from configs.lane_detection.common.optims.sgd006 import optimizer
from configs.lane_detection.common.optims.ep50_poly_warmup200 import lr_scheduler


train = dict(
    exp_name='mobilenetv3-large_resa_tusimple',
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
    original_size=(720, 1280),
    num_classes=7,
    num_epochs=50,
    collate_fn=None,  # 'dict_collate_fn' for LSTR
    seg=True  # Seg-based method or not
)

test = dict(
    exp_name='mobilenetv3-large_resa_tusimple',
    workers=10,
    batch_size=80,
    checkpoint='./checkpoints/mobilenetv3-large_resa_tusimple/model.pt',
    # Device args
    device='cuda',

    save_dir='./checkpoints',

    seg=True,
    gap=10,
    ppl=56,
    thresh=0.3,
    collate_fn=None,  # 'dict_collate_fn' for LSTR
    input_size=(360, 640),
    original_size=(720, 1280),
    max_lane=5,
    dataset_name='tusimple'
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
        num_classes=7
    ),
    reducer_cfg=dict(
        name='RESAReducer',
        in_channels=960,
        reduce=128
    ),
    lane_classifier_cfg=dict(
        name='EDLaneExist',
        num_output=7 - 1,
        flattened_size=1100,
        dropout=0.1,
        pool='avg'
    )
)
