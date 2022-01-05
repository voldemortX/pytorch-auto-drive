# Data pipeline
from configs.lane_detection.common.datasets.culane_seg import dataset
from configs.lane_detection.common.datasets.train_level0_288 import train_augmentation
from configs.lane_detection.common.datasets.test_288 import test_augmentation

# Optimization pipeline
from configs.lane_detection.common.optims.segloss_5class import loss
from configs.lane_detection.common.optims.adamw00006 import optimizer
# from configs.lane_detection.common.optims.ep12_poly_warmup200 import lr_scheduler

optimizer_decay0 = dict(
    filter_group=('absolute_pos_embed', 'relative_position_bias_table', 'norm')
)

lr_scheduler = dict(
    name='poly_scheduler_with_linearwarmup',
    epochs=10,
    power=1,
    warmup_steps=1500,
    warmup_ratio=1e-6,
)

