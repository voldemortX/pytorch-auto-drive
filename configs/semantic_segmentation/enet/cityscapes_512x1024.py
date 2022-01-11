# Data pipeline
from configs.semantic_segmentation.common.datasets.cityscapes import dataset
from configs.semantic_segmentation.common.datasets.city_train_half_512_wo_norm import train_augmentation
from configs.semantic_segmentation.common.datasets.city_test_half_wo_norm import test_augmentation

# Optimization pipeline
from configs.semantic_segmentation.common.optims.celoss_cityscapes_balanced import loss
from configs.semantic_segmentation.common.optims.adam00008_wd00002 import optimizer
from configs.semantic_segmentation.common.optims.ep300 import lr_scheduler


train = dict(
    exp_name='enet_cityscapes_512x1024',
    workers=8,
    batch_size=16,
    checkpoint=None,
    # Device args
    world_size=0,
    dist_url='env://',
    device='cuda',

    val_num_steps=1000,  # validation/checkpointing interval (steps)
    save_dir='./checkpoints',

    num_epochs=300,
    collate_fn=None,
    input_size=(512, 1024),
    original_size=(512, 1024),
    num_classes=19,

    # For selective evaluation (e.g., SYNTHIA selects 13/16 classes from Cityscapes)
    eval_classes=19,
    selector=None,

    # For ENet encoder pre-training
    encoder_only=False,
    encoder_size=None
)

test = dict(
    exp_name='enet_cityscapes_512x1024',
    workers=0,
    batch_size=1,
    checkpoint='./checkpoints/enet_cityscapes_512x1024/model.pt',
    # Device args
    device='cuda',

    save_dir='./checkpoints',

    collate_fn=None,  # 'dict_collate_fn' for LSTR
    original_size=(512, 1024),
    num_classes=19,

    # For selective evaluation (e.g., SYNTHIA selects 13/16 classes from Cityscapes)
    eval_classes=19,
    selector=None,

    # For ENet encoder pre-training
    encoder_only=False,
    encoder_size=None
)

model = dict(
    name='ENet',
    num_classes=19,
    encoder_relu=False,
    decoder_relu=True,
    dropout_1=0.01,
    dropout_2=0.1,
    encoder_only=False,
    pretrained_weights='./checkpoints/enet_cityscapes_512x1024_encoder/model.pt'
)
