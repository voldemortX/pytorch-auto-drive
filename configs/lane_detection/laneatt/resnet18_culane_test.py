# Data pipeline
# from configs.lane_detection.common.datasets.culane import dataset
# from configs.lane_detection.common.datasets.train_level0_288 import train_augmentation
from configs.lane_detection.common.datasets.test_288 import test_augmentation

# Optimization pipeline
from configs.lane_detection.common.optims.matchingloss_polynomial import loss
from configs.lane_detection.common.optims.adam000025 import optimizer
from configs.lane_detection.common.optims.ep150_step import lr_scheduler


dataset = dict(
    name='CULane',
    image_set='train',  # Only set for training. Testing will override this value by --state.
    root='/home/guoshaohua/dataset/culane',
    padding_mask=False,
    is_process=False,
)


train_augmentation = dict(
    name='Compose',
    transforms=[
        dict(
            name='RandomAffine',
            degrees=(-6, 6),
            translate=(25, 10),
            scale=(0.85, 1.15),
        ),
        dict(
            name='RandomHorizontalFlip',
            flip_prob=0.5
        ),
        dict(
            name='ToXOffset',
            num_points=72,
            image_size=(360, 640),
            max_lanes=4
        ),
        dict(
            name='ToTensor'
        ),
    ]
)
# ---- ignore ------
train = dict(
    exp_name='resnet18s_lstr_culane',
    workers=2,
    batch_size=4,
    checkpoint=None,
    # Device args
    world_size=0,
    dist_url='env://',
    device='cuda',

    val_num_steps=0,  # Seg IoU validation (mostly useless)
    save_dir='./checkpoints',

    input_size=(288, 800),
    original_size=(590, 1640),
    num_classes=None,
    num_epochs=12,
    collate_fn=None,  # 'dict_collate_fn' for LSTR
    seg=False,  # Seg-based method or not
)

test = dict(
    exp_name='resnet18s_lstr_culane',
    workers=10,
    batch_size=80,
    checkpoint='./checkpoints/resnet18s_lstr_culane/model.pt',
    # Device args
    device='cuda',

    save_dir='./checkpoints',

    seg=False,
    gap=20,
    ppl=18,
    thresh=None,
    collate_fn='dict_collate_fn',  # 'dict_collate_fn' for LSTR
    input_size=(288, 800),
    original_size=(590, 1640),
    max_lane=4,
    dataset_name='culane'
)

model = dict(
    name='LSTR',
    expansion=2,
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
        expansion=2
    )
)