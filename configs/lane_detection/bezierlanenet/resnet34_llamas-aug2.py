# Data pipeline
from configs.lane_detection.common.datasets.llamas_bezier import dataset
from configs.lane_detection.common.datasets.train_level2_360 import train_augmentation
from configs.lane_detection.common.datasets.test_360 import test_augmentation

# Optimization pipeline
from configs.lane_detection.common.optims.matchingloss_bezier import loss
from configs.lane_detection.common.optims.adam00006_dcn import optimizer
from configs.lane_detection.common.optims.ep20_cosine import lr_scheduler


train = dict(
    exp_name='resnet34_bezierlanenet_llamas-aug2',
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
    num_classes=None,
    num_epochs=20,
    collate_fn='dict_collate_fn',  # 'dict_collate_fn' for LSTR
    seg=False,  # Seg-based method or not
)

test = dict(
    exp_name='resnet34_bezierlanenet_llamas-aug2',
    workers=0,
    batch_size=1,
    checkpoint='./checkpoints/resnet34_bezierlanenet_llamas-aug2/model.pt',
    # Device args
    device='cuda',

    save_dir='./checkpoints',

    seg=False,
    gap=1,
    ppl=417,
    thresh=0.3,
    collate_fn='dict_collate_fn',  # 'dict_collate_fn' for LSTR
    input_size=(360, 640),
    original_size=(717, 1276),
    max_lane=4,
    dataset_name='llamas'
)

model = dict(
    name='BezierLaneNet',
    image_height=288,
    num_regression_parameters=8,  # 3 x 2 + 2 = 8 (Cubic Bezier Curve)

    # Inference parameters
    thresh=0.95,
    local_maximum_window_size=9,

    # Backbone (3-stage resnet (no dilation) + 2 extra dilated blocks)
    backbone_cfg=dict(
        name='predefined_resnet_backbone',
        backbone_name='resnet34',
        return_layer='layer3',
        pretrained=True,
        replace_stride_with_dilation=[False, False, False]
    ),
    reducer_cfg=None,  # No need here
    dilated_blocks_cfg=dict(
        name='predefined_dilated_blocks',
        in_channels=256,
        mid_channels=64,
        dilations=[4, 8]
    ),

    # Head, Fusion module
    feature_fusion_cfg=dict(
        name='FeatureFlipFusion',
        channels=256
    ),
    head_cfg=dict(
        name='ConvProjection_1D',
        num_layers=2,
        in_channels=256,
        bias=True,
        k=3
    ),  # Just some transforms of feature, similar to FCOS heads, but shared between cls & reg branches

    # Auxiliary binary segmentation head (automatically discarded in eval() mode)
    aux_seg_head_cfg=dict(
        name='SimpleSegHead',
        in_channels=256,
        mid_channels=64,
        num_classes=1
    )
)
