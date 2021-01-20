import os
import numpy as np
import imageio
from PIL import Image
from tqdm import tqdm
from base_dirs import base_synthia


for image_set in ['train', 'val']:
    mask_dir = os.path.join(base_synthia, 'GT/LABELS', image_set)
    new_mask_dir = os.path.join(base_synthia, 'GT/LABELS_CONVERTED', image_set)
    if not os.path.exists(new_mask_dir):
        os.makedirs(new_mask_dir)
    splits_file = os.path.join(base_synthia, 'data_lists', image_set + '.txt')
    with open(splits_file, 'r') as f:
        file_names = [x.strip() for x in f.readlines()]

    mask_files = [os.path.join(mask_dir, x + '.png') for x in file_names]
    new_mask_files = [os.path.join(new_mask_dir, x + '.png') for x in file_names]
    for index in tqdm(range(len(mask_files))):
        mask = np.asarray(imageio.imread(mask_files[index], format='PNG-FI'))[:, :, 0]
        im = Image.fromarray(np.uint8(mask))
        im.save(new_mask_files[index])
