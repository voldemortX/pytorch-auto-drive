# This script generates Cityscapes official demos from downloaded demo images
import os
from utils.frames_to_video import frames2video

from importmagician import import_from
with import_from('./'):
    from configs.semantic_segmentation.common.datasets._utils import CITYSCAPES_ROOT
base = os.path.join(CITYSCAPES_ROOT, 'all_demoVideo', 'leftImg8bit', 'demoVideo')

# Get image file list
sub_dirs = sorted(os.listdir(base))

for sub_dir in sub_dirs:
    filenames = sorted(os.listdir(os.path.join(base, sub_dir)))

    # Save video
    frames2video(os.path.join(base, sub_dir), sub_dir + '.avi', filenames, fps=17)
