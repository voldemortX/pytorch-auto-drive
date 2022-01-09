# Data pipeline
from configs.lane_detection.common.datasets.culane_seg import dataset
from configs.lane_detection.common.datasets.train_level0_288 import train_augmentation
from configs.lane_detection.common.datasets.test_288 import test_augmentation

# Optimization pipeline
from configs.lane_detection.common.optims.segloss_5class import loss
from configs.lane_detection.common.optims.sgd02 import optimizer
from configs.lane_detection.common.optims.ep12_poly_warmup200 import lr_scheduler


train_args_default = dict(
    exp_name='mobilenetv2_baseline_culane',
    workers=10,
    batch_size=20,
    checkpoint=None,
    # Device args
    world_size=0,
    dist_url='env://',
    device='cuda',

    val_num_steps=0,  # Seg IoU validation (mostly useless)
    save_dir='./checkpoints'
)
test_args_default = dict(
    exp_name='mobilenetv2_baseline_culane',
    workers=10,
    batch_size=80,
    checkpoint='./checkpoints/mobilenetv2_baseline_culane/model.pt',
    # Device args
    device='cuda',

    save_dir='./checkpoints'
)

# Configs
train = dict(
    input_size=(288, 800),
    original_size=(590, 1640),
    num_classes=5,
    num_epochs=12,
    collate_fn=None,  # 'dict_collate_fn' for LSTR
    seg=True  # Seg-based method or not
)

test = dict(
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
    name='SegRepVGG',
    backbone_cfg=dict(
        name='MobileNetV2',
        pretrained='https://download.pytorch.org/models/mobilenet_v2-b0353104.pth',
        widen_factor=1,
        strides=(1, 2, 2, 1, 1, 1, 1),
        dilations=(1, 1, 1, 2, 2, 4, 4),
        out_indices=(1, 2, 4, 6),
        out_stride=8,
    ),
    classifier_cfg=dict(
        name='DeepLabV1Head',
        in_channels=96,
        num_classes=5,
        dilation=1
    ),
    lane_classifier_cfg=dict(
        name='SimpleLaneExist',
        num_output=5 - 1,
        flattened_size=4500,
    ),
    spatial_conv_cfg=dict(
        name='SpatialConv',
        num_channels=96
    ),
)
