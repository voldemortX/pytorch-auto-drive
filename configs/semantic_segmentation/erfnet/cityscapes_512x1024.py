# Data pipeline
from configs.semantic_segmentation.common.datasets.cityscapes import dataset
from configs.semantic_segmentation.common.datasets.city_train_half_512_wo_norm import train_augmentation
from configs.semantic_segmentation.common.datasets.city_test_half_wo_norm import test_augmentation

# Optimization pipeline
from configs.semantic_segmentation.common.optims.celoss_cityscapes_balanced import loss
from configs.semantic_segmentation.common.optims.adam00007 import optimizer
from configs.semantic_segmentation.common.optims.ep150_epoch import lr_scheduler


train = dict(
    exp_name='erfnet_cityscapes_512x1024',
    workers=8,
    batch_size=10,
    checkpoint=None,
    # Device args
    world_size=0,
    dist_url='env://',
    device='cuda',

    val_num_steps=1000,  # validation/checkpointing interval (steps)
    save_dir='./checkpoints',

    num_epochs=150,
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
    exp_name='erfnet_cityscapes_512x1024',
    workers=0,
    batch_size=1,
    checkpoint='./checkpoints/erfnet_cityscapes_512x1024/model.pt',
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
    name='ERFNet',
    num_classes=19,
    dropout_1=0.03,
    dropout_2=0.3,
    pretrained_weights='erfnet_encoder_pretrained.pth.tar'
)
