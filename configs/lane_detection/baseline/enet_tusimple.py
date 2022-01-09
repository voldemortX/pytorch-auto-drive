# Data pipeline
from configs.lane_detection.common.datasets.tusimple_seg import dataset
from configs.lane_detection.common.datasets.train_level0_360 import train_augmentation
from configs.lane_detection.common.datasets.test_360 import test_augmentation

# Optimization pipeline
from configs.lane_detection.common.optims.segloss_7class import loss
from configs.lane_detection.common.optims.sgd04 import optimizer
from configs.lane_detection.common.optims.ep50_poly_warmup200 import lr_scheduler


train_args_default = dict(
    exp_name='enet_baseline_tusimple',
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
    exp_name='enet_baseline_tusimple',
    workers=10,
    batch_size=80,
    checkpoint='./checkpoints/enet_baseline_tusimple/model.pt',
    # Device args
    device='cuda',

    save_dir='./checkpoints'
)

# Configs
train = dict(
    input_size=(360, 640),
    original_size=(720, 1280),
    num_classes=7,
    num_epochs=50,
    collate_fn=None,  # 'dict_collate_fn' for LSTR
    seg=True  # Seg-based method or not
)

test = dict(
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
    name='ENet',
    num_classes=7,
    encoder_relu=False,
    decoder_relu=True,
    dropout_1=0.01,
    dropout_2=0.1,
    encoder_only=False,
    pretrained_weights=None,
    lane_classifier_cfg=dict(
        name='EDLaneExist',
        num_output=7 - 1,
        flattened_size=4400,
        dropout=0.1,
        pool='avg'
    )
)
