# This script converts TuSimple validation set (0531) to a video
import os
import yaml
import cv2
from cv2 import VideoWriter_fourcc
from mmcv.utils import check_file_exist, mkdir_or_exist, scandir, track_progress


# Modified from open-mmlab/mmcv
def frames2video(frame_dir,
                 video_file,
                 file_names,
                 fps=20,
                 fourcc='XVID',
                 show_progress=True):
    """Read the frame images from a directory and join them as a video.
    Args:
        frame_dir (str): The directory containing video frames.
        video_file (str): Output filename.
        file_names (list[str]): Image files
        fps (float): FPS of the output video.
        fourcc (str): Fourcc of the output video, this should be compatible
            with the output file type.
        show_progress (bool): Whether to show a progress bar.
    """

    first_file = os.path.join(frame_dir, file_names[0])
    check_file_exist(first_file, 'The start frame not found: ' + first_file)
    img = cv2.imread(first_file)
    height, width = img.shape[:2]
    resolution = (width, height)
    vwriter = cv2.VideoWriter(video_file, VideoWriter_fourcc(*fourcc), fps, resolution)

    def write_frame(file_idx):
        filename = os.path.join(frame_dir, file_names[file_idx])
        img = cv2.imread(filename)
        vwriter.write(img)

    if show_progress:
        track_progress(write_frame, range(len(file_names)))
    else:
        for i in range(len(file_names)):
            write_frame(i)
    vwriter.release()


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
