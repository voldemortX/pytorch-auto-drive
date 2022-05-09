from importmagician import import_from
with import_from('./'):
    # Data pipeline
    from configs.lane_detection.common.datasets.culane_seg import dataset
    from configs.lane_detection.common.datasets.train_level0_288 import train_augmentation
    from configs.lane_detection.common.datasets.test_288 import test_augmentation

    # Optimization pipeline
    from configs.lane_detection.common.optims.segloss_5class import loss
    from configs.lane_detection.common.optims.sgd02 import optimizer
    from configs.lane_detection.common.optims.ep12_poly_warmup200 import lr_scheduler


train = dict(
    exp_name='repvgg-a1_baseline_culane',
    workers=5,
    batch_size=10,
    checkpoint=None,
    # Device args
    world_size=2,
    dist_url='tcp://localhost:12345',
    device='cuda',
    val_num_steps=0,  # Seg IoU validation (mostly useless)
    save_dir='./checkpoints',

    input_size=(288, 800),
    original_size=(590, 1640),
    num_classes=5,
    num_epochs=12,
    collate_fn=None,  # 'dict_collate_fn' for LSTR
    seg=True  # Seg-based method or not
)

test = dict(
    exp_name='repvgg-a1_baseline_culane',
    workers=10,
    batch_size=80,
    checkpoint='./checkpoints/repvgg-a1_baseline_culane/model.pt',
    # Device args
    device='cuda',
    save_dir='./checkpoints',

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

model = dict(
    name='SegRepVGG',
    # num_classes=5,
    # dropout_1=0.1,
    backbone_cfg=dict(
        name='RepVggEncoder',
        backbone_name='RepVGG-A1',
        pretrained=True,
        deploy=False
    ),
    reducer_cfg=dict(
        name='RESAReducer',
        in_channels=1280,
        reduce=128
    ),
    lane_classifier_cfg=dict(
        name='SimpleLaneExist',
        num_output=5 - 1,
        flattened_size=4500,
    ),
    classifier_cfg=dict(
        name='DeepLabV1Head',
        in_channels=128,
        num_classes=5,
        dilation=1
    )
)
