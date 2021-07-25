# This script converts TuSimple validation set (0531) to a video
import os
import yaml
from frames_to_video import frames2video


with open('configs.yaml', 'r') as f:  # Safer and cleaner than box/EasyDict
    configs = yaml.load(f, Loader=yaml.Loader)
base = os.path.join(configs['TUSIMPLE']['BASE_DIR'], 'clips', '0601')

# Get image file list
sub_dirs = sorted(os.listdir(base))
file_names = []
for sub_dir in sub_dirs:
    for i in range(20):
        file_names.append(os.path.join(sub_dir, str(i + 1) + '.jpg'))

# Save video
# frames2video(base, 'tusimple_val.avi', file_names)
frames2video(base, 'tusimple_val_1min.avi', file_names[:1200])  # 1 minute
