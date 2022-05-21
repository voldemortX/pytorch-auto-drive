# Data pipeline
# from configs.lane_detection.common.datasets.culane import dataset
# from configs.lane_detection.common.datasets.train_level0_288 import train_augmentation
# from configs.lane_detection.common.datasets.test_288 import test_augmentation

# Optimization pipeline
# from configs.lane_detection.common.optims.matchingloss_polynomial import loss
# from configs.lane_detection.common.optims.adam000025 import optimizer
# from configs.lane_detection.common.optims.ep150_step import lr_scheduler

lr_scheduler = dict(
    name='CosineAnnealingLRWrapper',
    epochs=15,
)

from configs.lane_detection.common.datasets._utils import TUSIMPLE_ROOT
dataset = dict(
    name='TuSimple',
    image_set='train',  # Only set for training. Testing will override this value by --state.
    root=TUSIMPLE_ROOT,
    padding_mask=False,
    is_process=False,
)

optimizer = dict(
    name='torch_optimizer',
    torch_optim_class='Adam',
    lr=0.0003
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
            name='Resize',
            size_image=(360, 640),
            size_label=(360, 640)
        ),
        dict(
            name='LaneATTLabelFormat',
            num_points=72,
            image_size=(360, 640),
            max_lanes=5
        ),
        dict(
            name='ToTensor'  # div 255 ???
        ),
    ]
)


test_augmentation = dict(
    name='Compose',
    transforms=[
        dict(
            name='Resize',
            size_image=(360, 640),
            size_label=(360, 640)
        ),
        dict(
            name='ToTensor',
            reverse_channels=True,
        ),
    ]
)


loss = dict(
    name='LaneAttLoss',
    cls_weight=10.,
    reg_weight=1.,
    alpha=0.25,
    gamma=2.,
    num_offsets=72,
    num_strips=72 - 1,
    t_pos=15.,
    t_neg=20.,
    reduction='mean'
)

train = dict(
    exp_name='resnet18_laneatt_tusimple_1',
    workers=4,
    batch_size=8,
    checkpoint=None,
    # Device args
    world_size=0,
    dist_url='env://',
    device='cuda',

    val_num_steps=0,  # Seg IoU validation (mostly useless)
    save_dir='./checkpoints',

    input_size=(360, 640),
    original_size=(720, 1280),
    num_classes=None,
    num_epochs=100,
    collate_fn='dict_collate_fn',  # 'dict_collate_fn' for LSTR
    seg=False,  # Seg-based method or not
)

test = dict(
    exp_name='resnet18_laneatt_tusimple_1',
    workers=4,
    batch_size=32,
    checkpoint='./checkpoints/resnet18_laneatt_tusimple_1/model_0100.pt',
    # Device args
    device='cuda',
    save_dir='./checkpoints',
    seg=False,
    gap=10,
    ppl=56,
    thresh=None,
    collate_fn=None,  # 'dict_collate_fn' for LSTR
    input_size=(360, 640),
    original_size=(720, 1280),
    max_lane=5,
    dataset_name='tusimple'
)

model = dict(
    name='LaneAtt',
    backbone_channels=512,
    backbone_os=32,
    num_points=72,
    img_w=640,
    img_h=360,
    topk_anchors=1000,
    anchor_freq_path='tusimple_anchors_freq.pt',
    anchor_feat_channels=64,
    # nms config
    conf_thres=0.2,
    nms_thres=45,
    nms_topk=5,  # max # lanes of the dataset
    # backbone config
    backbone_cfg=dict(
        name='predefined_resnet_backbone',
        backbone_name='resnet18',
        return_layer='layer4',
        pretrained=True,
        replace_stride_with_dilation=[False, False, False]
    )
)
