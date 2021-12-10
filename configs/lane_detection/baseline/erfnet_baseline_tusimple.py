# Dicts to become configs
__all__ = ['train',
           'test',
           'model',
           'train_augmentation',
           'test_augmentation',
           'lr_scheduler',
           'optimizer',
           'loss',
           'dataset']

# Default args that can be overridden in commandline
exp_name = 'erfnet_baseline_tusimple'
train_args_default = dict(
    exp_name=exp_name,
    workers=10,
    batch_size=20,
    mixed_precision=True,
    continue_from=None,
    state=0,
    # Device args
    world_size=0,
    dist_url='env://',
    device='cuda'
)
test_args_default = dict(
    exp_name=exp_name,
    workers=10,
    batch_size=80,
    mixed_precision=True,
    continue_from=exp_name + '.pt',
    state=2,
    # Device args
    device='cuda'
)

# Common statics
input_size = (360, 640)
original_size = (720, 1280)
num_classes = 7
dataset_name = 'tusimple'
dataset_root = '../../dataset/tusimple'  # path relative to main_*.py
max_lane = 5  # for lane pruning
gap = 10
ppl = 56
thresh = 0.3
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
epochs = 50

# Configs
train = dict(
    input_size=input_size,
    original_size=original_size,
    num_classes=num_classes,
    num_epochs=epochs,
    collate_fn=None,  # 'dict_collate_fn' for LSTR
    seg=True,  # Seg-based method or not
    validation=False,  # Seg IoU validation (mostly useless)
    val_num_steps=0,
)
train.update(train_args_default)

test = dict(
    seg=True,
    gap=gap,
    ppl=ppl,
    thresh=thresh,
    collate_fn=None,  # 'dict_collate_fn' for LSTR
    input_size=input_size,
    original_size=original_size,
    max_lane=max_lane,
    dataset_name=dataset_name
)
test.update(test_args_default)

model = dict(
    name='ERFNet',
    num_classes=num_classes,
    dropout_1=0.3,
    dropout_2=0.3,
    pretrained_weights='erfnet_encoder_pretrained.pth.tar',
    lane_classifier_cfg=dict(
        name='EDLaneExist',
        num_output=num_classes - 1,
        flattened_size=4400,
        dropout=0.3,
        pool='max'
    )
)

# Aug level: simple
train_augmentation = dict(
    name='Compose',
    transforms=[
        dict(
            name='Resize',
            size_image=input_size,
            size_label=input_size
        ),
        dict(
            name='RandomRotation',
            degrees=3
        ),
        dict(
            name='ToTensor'
        ),
        dict(
            name='Normalize',
            mean=mean,
            std=std
        )
    ]
)

test_augmentation = dict(
    name='Compose',
    transforms=[
        dict(
            name='Resize',
            size_image=input_size,
            size_label=input_size
        ),
        dict(
            name='ToTensor'
        ),
        dict(
            name='Normalize',
            mean=mean,
            std=std
        )
    ]
)

lr_scheduler = dict(
    name='poly_scheduler_with_warmup',
    epochs=epochs,
    power=0.9,
    warmup_steps=200
)

optimizer = dict(
    name='torch_optimizer',
    torch_optim_class='SGD',
    lr=0.2,
    momentum=0.9,
    weight_decay=1e-4
)

loss = dict(
    name='LaneLoss',
    existence_weight=0.1,
    ignore_index=255,
    weight=[0.4, 1, 1, 1, 1, 1, 1]
)

dataset = dict(
    name='TuSimpleAsSegmentation',
    image_set='train',  # Only set for training. Testing will override this value by --state.
    root=dataset_root
)
