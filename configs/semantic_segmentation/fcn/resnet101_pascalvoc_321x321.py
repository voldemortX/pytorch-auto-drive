# Data pipeline
from configs.semantic_segmentation.common.datasets.pascal_voc import dataset
from configs.semantic_segmentation.common.datasets.voc_train_321 import train_augmentation
from configs.semantic_segmentation.common.datasets.voc_test_505 import test_augmentation

# Optimization pipeline
from configs.semantic_segmentation.common.optims.celoss import loss
from configs.semantic_segmentation.common.optims.sgd0002 import optimizer
from configs.semantic_segmentation.common.optims.ep30 import lr_scheduler


train = dict(
    exp_name='resnet101_fcn_pascalvoc_321x321',
    workers=4,
    batch_size=8,
    checkpoint=None,
    # Device args
    world_size=0,
    dist_url='env://',
    device='cuda',

    val_num_steps=1000,  # validation/checkpointing interval (steps)
    save_dir='./checkpoints',

    num_epochs=30,
    collate_fn=None,
    input_size=(321, 321),
    original_size=(505, 505),
    num_classes=21,

    # For selective evaluation (e.g., SYNTHIA selects 13/16 classes from Cityscapes)
    eval_classes=21,
    selector=None,

    # For ENet encoder pre-training
    encoder_only=False,
    encoder_size=None
)

test = dict(
    exp_name='resnet101_fcn_pascalvoc_321x321',
    workers=0,
    batch_size=1,
    checkpoint='./checkpoints/resnet101_fcn_pascalvoc_321x321/model.pt',
    # Device args
    device='cuda',

    save_dir='./checkpoints',

    collate_fn=None,  # 'dict_collate_fn' for LSTR
    original_size=(505, 505),
    num_classes=21,

    # For selective evaluation (e.g., SYNTHIA selects 13/16 classes from Cityscapes)
    eval_classes=21,
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
        name='FCNHead',
        in_channels=2048,
        num_classes=21
    )
)
