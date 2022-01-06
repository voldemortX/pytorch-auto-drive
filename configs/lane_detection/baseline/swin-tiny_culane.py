# Data pipeline
from configs.lane_detection.common.datasets.culane_seg import dataset
from configs.lane_detection.common.datasets.train_level0_288 import train_augmentation
from configs.lane_detection.common.datasets.test_288 import test_augmentation

# Optimization pipeline
from configs.lane_detection.common.optims.segloss_5class import loss
from configs.lane_detection.common.optims.adamw00006_swin import optimizer
# from configs.lane_detection.common.optims.ep12_poly_warmup200 import lr_scheduler

lr_scheduler = dict(
    name='poly_scheduler_with_warmup',
    epochs=10,
    power=1,  # ? Kept for consistency with official repo
    warmup_steps=1500,
    start_lr_ratio=1e-6,
)
