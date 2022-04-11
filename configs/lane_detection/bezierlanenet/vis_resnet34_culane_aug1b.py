# 1. Import from the corresponding config
# Or you can just copy-paste (if your config filename includes - or something)
from configs.lane_detection.bezierlanenet.resnet34_culane_aug1b import *

# 2. Define vis_dataset
from configs.lane_detection.common.datasets._utils import CULANE_ROOT

vis_dataset = dict(
    name='CULaneVis',
    root_dataset=CULANE_ROOT,
    root_output='./test_culane_vis',
    root_keypoint=None,
    image_set='val'
)

# 3. You can optionally overwrite things, like
# test['checkpoint'] = './resnet34_bezierlanenet_culane_aug2_20211109.pt'
