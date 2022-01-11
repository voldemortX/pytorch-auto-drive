# Data pipeline
from configs.lane_detection.common.datasets.culane_seg import dataset
from configs.lane_detection.common.datasets.train_level0_288 import train_augmentation
from configs.lane_detection.common.datasets.test_288 import test_augmentation

# Optimization pipeline
from configs.lane_detection.common.optims.segloss_5class import loss
from configs.lane_detection.common.optims.sgd008 import optimizer
from configs.lane_detection.common.optims.ep12_poly_warmup500 import lr_scheduler

# Default args that can be overridden in commandline
train_args_default = dict(
    exp_name='resnet34_naive_baseline_culane',
    workers=5,
    batch_size=10,
    checkpoint=None,
    # Device args
    world_size=2,
    dist_url='tcp://localhost:12345',
    device='cuda',

    val_num_steps=0,  # Seg IoU validation (mostly useless)
    save_dir='./checkpoints'
)
test_args_default = dict(
    exp_name='resnet34_baseline_culane',
    workers=4,
    batch_size=32,
    checkpoint='./checkpoints/resnet50_baseline_culane/model.pt',
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
    seg=True,  # Seg-based method or not
)
train.update(train_args_default)

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
test.update(test_args_default)

# Essentially DeepLabV1 without dilation like in SCNN paper
model = dict(
    name='DLabV1AugLane',
    backbone_cfg=dict(
        name='predefined_resnet_backbone',
        backbone_name='resnet34',
        return_layer={'layer4': 'out4', 'layer3': 'out3', 'layer2': 'out2'},
        pretrained=True,
        replace_stride_with_dilation=[False, False, False]
    ),
    # reducer_cfg=dict(
    #     name='RESAReducer',
    #     in_channels=245,
    #     reduce=128
    # ),
    classifier_cfg=dict(
        name='DeepLabV1Head',
        in_channels=128,
        num_classes=5,
        dilation=1
    ),
    lane_classifier_cfg=dict(
        name='SimpleLaneExist',
        num_output=5 - 1,
        flattened_size=4500
    ),
    aug_head_cfg=dict(
        name='FPNHead',
        in_channels=[128, 256, 512],
        channels=128,
        input_induces=None,
        align_corners=False,
        is_dict=True,
        fusion='add'
    )
)
