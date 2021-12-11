# Data pipeline
from configs.lane_detection.common.datasets.tusimple_seg import dataset
from configs.lane_detection.common.datasets.seg_level0_360 import train_augmentation
from configs.lane_detection.common.datasets.seg_test_360 import test_augmentation

# Optimization pipeline
from configs.lane_detection.common.optims.segloss_7class import loss
from configs.lane_detection.common.optims.sgd02 import optimizer
from configs.lane_detection.common.optims.ep50_warmup200 import lr_scheduler

# Default args that can be overridden in commandline
train_args_default = dict(
    exp_name='erfnet_baseline_tusimple',
    workers=10,
    batch_size=20,
    continue_from=None,
    # Device args
    world_size=0,
    dist_url='env://',
    device='cuda'
)
test_args_default = dict(
    exp_name='erfnet_baseline_tusimple',
    workers=10,
    batch_size=80,
    continue_from='erfnet_baseline_tusimple.pt',
    # Device args
    device='cuda'
)

# Configs
train = dict(
    input_size=(360, 640),
    original_size=(720, 1280),
    num_classes=7,
    num_epochs=50,
    collate_fn=None,  # 'dict_collate_fn' for LSTR
    seg=True,  # Seg-based method or not
    validation=False,  # Seg IoU validation (mostly useless)
    val_num_steps=0,
)
train.update(train_args_default)

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
test.update(test_args_default)

model = dict(
    name='ERFNet',
    num_classes=7,
    dropout_1=0.3,
    dropout_2=0.3,
    pretrained_weights='erfnet_encoder_pretrained.pth.tar',
    lane_classifier_cfg=dict(
        name='EDLaneExist',
        num_output=7 - 1,
        flattened_size=4400,
        dropout=0.3,
        pool='max'
    )
)
