# Data pipeline
from configs.semantic_segmentation.common.datasets.gtav import dataset
from configs.semantic_segmentation.common.datasets.gtav_train_hd_512 import train_augmentation
from configs.semantic_segmentation.common.datasets.city_test_hd import test_augmentation

# For UDA baseline setting (train on GTAV, test on Cityscapes)
from configs.semantic_segmentation.common.datasets.cityscapes import dataset as test_dataset

# Optimization pipeline
from configs.semantic_segmentation.common.optims.celoss import loss
from configs.semantic_segmentation.common.optims.sgd0002 import optimizer
from configs.semantic_segmentation.common.optims.ep10 import lr_scheduler


train_args_default = dict(
    exp_name='resnet101_deeplabv2_gtav_512x1024',
    workers=8,
    batch_size=4,
    checkpoint=None,
    # Device args
    world_size=0,
    dist_url='env://',
    device='cuda',

    val_num_steps=1000,  # validation/checkpointing interval (steps)
    save_dir='./checkpoints'
)
test_args_default = dict(
    exp_name='resnet101_deeplabv2_gtav_512x1024',
    workers=0,
    batch_size=1,
    checkpoint='./checkpoints/resnet101_deeplabv2_gtav_512x1024/model.pt',
    # Device args
    device='cuda',

    save_dir='./checkpoints'
)

# Configs
train = dict(
    num_epochs=10,
    collate_fn=None,
    input_size=(512, 1024),
    original_size=(1024, 2048),
    num_classes=19,

    # For selective evaluation (e.g., SYNTHIA selects 13/16 classes from Cityscapes)
    eval_classes=19,
    selector=None,

    # For ENet encoder pre-training
    encoder_only=False,
    encoder_size=None,

    # To avoid OOM
    val_batch_size=2
)

test = dict(
    collate_fn=None,  # 'dict_collate_fn' for LSTR
    original_size=(1024, 2048),
    num_classes=19,

    # For selective evaluation (e.g., SYNTHIA selects 13/16 classes from Cityscapes)
    eval_classes=19,
    selector=None,

    # For ENet encoder pre-training
    encoder_only=False,
    encoder_size=None
)

model = dict(
    name='standard_segmentation_model',
    pretrained=False,  # MSCOCO pre-training, not recommended (not the same COCO pre-training in DeepLab paper)
    backbone_cfg=dict(
        name='predefined_resnet_backbone',
        backbone_name='resnet101',
        return_layer='layer4',
        pretrained=True,  # ImageNet pre-training
        replace_stride_with_dilation=[False, True, True]
    ),
    classifier_cfg=dict(
        name='DeepLabV2Head',
        in_channels=2048,
        num_classes=19
    )
)
