# Data pipeline
from configs.lane_detection.common.datasets.culane_seg import dataset
from configs.lane_detection.common.datasets.train_level0_288 import train_augmentation
from configs.lane_detection.common.datasets.test_288 import test_augmentation

# Optimization pipeline
from configs.lane_detection.common.optims.segloss_5class import loss
from configs.lane_detection.common.optims.adamw00006_swin import optimizer
# from configs.lane_detection.common.optims.ep12_poly_warmup200 import lr_scheduler

lr_scheduler = dict(
    name='poly_scheduler_with_warmup',
    epochs=36,
    power=1,  # ? Kept for consistency with official repo
    warmup_steps=1500,
    start_lr_ratio=1e-6,
)

# Default args that can be overridden in commandline
train_args_default = dict(
    exp_name='swin-tiny_baseline_culane',
    workers=4,
    batch_size=8,
    checkpoint=None,
    # Device args
    world_size=2,
    dist_url='tcp://localhost:12345',
    device='cuda',
    val_num_steps=0,  # Seg IoU validation (mostly useless)
    save_dir='./checkpoints'
)
test_args_default = dict(
    exp_name='swin-tiny_baseline_culane',
    workers=4,
    batch_size=32,
    checkpoint='./checkpoints/swin-tiny_baseline_culane/model.pt',
    # Device args
    device='cuda',
    save_dir='./checkpoints'
)

# Configs
train = dict(
    input_size=(288, 800),
    original_size=(590, 1640),
    num_classes=5,
    num_epochs=36,
    collate_fn=None,  # 'dict_collate_fn' for LSTR
    seg=True  # Seg-based method or not
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

model = dict(
    name='DeepLabV1Lane',
    backbone_cfg=dict(
        name='SwinTransformer',
        pretrain_img_size=(288, 800),
        patch_size=4,
        in_chans=3,
        embed_dim=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=7,
        mlp_ratio=4.,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.3,
        # norm_layer=nn.LayerNorm,
        ape=False,
        patch_norm=True,
        out_indices=(0, 1, 2, 3),
        frozen_stages=-1,
        use_checkpoint=False,
        pretrained='swin_tiny_patch4_window7_224.pth',
        chosen_stages=2,
    ),
    reducer_cfg=dict(
        name='RESAReducer',
        in_channels=384,
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
        flattened_size=1125,
    )
)
