# This script converts TuSimple validation set (0531) to a video
import os
from utils.frames_to_video import frames2video

from importmagician import import_from
with import_from('./'):
    from configs.lane_detection.common.datasets._utils import TUSIMPLE_ROOT
base = os.path.join(TUSIMPLE_ROOT, 'clips', '0531')

# Get image file list
sub_dirs = sorted(os.listdir(base))
file_names = []
for sub_dir in sub_dirs:
    for i in range(20):
        file_names.append(os.path.join(sub_dir, str(i + 1) + '.jpg'))

# Save video
# frames2video(base, 'tusimple_val.avi', file_names)
frames2video(base, 'tusimple_val_1min.avi', file_names[:1200])  # 1 minute
