# This script generates Cityscapes official demos from downloaded demo images
import os
import yaml
from frames_to_video import frames2video


with open('configs.yaml', 'r') as f:  # Safer and cleaner than box/EasyDict
    configs = yaml.load(f, Loader=yaml.Loader)
base = os.path.join(configs['CITYSCAPES']['BASE_DIR'], 'all_demoVideo', 'leftImg8bit', 'demoVideo')

# Get image file list
sub_dirs = [os.path.join(base, sub_dir) for sub_dir in sorted(os.listdir(base))]

for sub_dir in sub_dirs:
    filenames = sorted(os.listdir(sub_dir))

    # Save video
    frames2video(base, sub_dir + '.avi', filenames, fps=30)
