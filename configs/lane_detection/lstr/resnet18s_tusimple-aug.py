# Data pipeline
from configs.lane_detection.common.datasets.tusimple import dataset
from configs.lane_detection.common.datasets.train_level1_360 import train_augmentation
from configs.lane_detection.common.datasets.test_360 import test_augmentation

# Optimization pipeline
from configs.lane_detection.common.optims.matchingloss_polynomial import loss
from configs.lane_detection.common.optims.adam000025 import optimizer
from configs.lane_detection.common.optims.ep2000_step import lr_scheduler


train_args_default = dict(
    exp_name='resnet18s_lstr-aug_tusimple',
    workers=16,
    batch_size=20,
    checkpoint=None,
    # Device args
    world_size=0,
    dist_url='env://',
    device='cuda',

    val_num_steps=0,  # >0 not supported
    save_dir='./checkpoints'
)
test_args_default = dict(
    exp_name='resnet18s_lstr-aug_tusimple',
    workers=10,
    batch_size=80,
    checkpoint='./checkpoints/resnet18s_lstr-aug_tusimple/model.pt',
    # Device args
    device='cuda',

    save_dir='./checkpoints'
)

# Configs
train = dict(
    seg=False,  # Seg-based method or not
    input_size=(360, 640),
    original_size=(720, 1280),
    num_classes=None,
    num_epochs=2000,
    collate_fn='dict_collate_fn'
)

test = dict(
    seg=False,
    gap=10,
    ppl=56,
    thresh=None,
    collate_fn='dict_collate_fn',
    input_size=(360, 640),
    original_size=(720, 1280),
    max_lane=5,
    dataset_name='tusimple'
)

model = dict(
    name='LSTR',
    expansion=1,
    num_queries=7,
    aux_loss=True,
    pos_type='sine',
    drop_out=0.1,
    num_heads=2,
    enc_layers=2,
    dec_layers=2,
    pre_norm=False,
    return_intermediate=True,
    lsp_dim=8,
    mlp_layers=3,
    thresh=0.5,
    backbone_cfg=dict(
        name='predefined_resnet_backbone',
        backbone_name='resnet18_reduced',
        return_layer='layer4',
        pretrained=False,
        replace_stride_with_dilation=[False, False, False],
        expansion=1
    )
)
