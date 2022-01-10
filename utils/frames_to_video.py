# Modified from open-mmlab/mmcv
import os
import cv2
from cv2 import VideoWriter_fourcc
from mmcv.utils import check_file_exist, track_progress


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
