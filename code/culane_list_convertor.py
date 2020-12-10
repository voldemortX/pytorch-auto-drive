# CULane (official lists)
# /driver_23_30frame/05151649_0422.MP4/00000.jpg /laneseg_label_w16/driver_23_30frame/05151649_0422.MP4/00000.png 1 1 1 1 =>
# /driver_23_30frame/05151649_0422.MP4/00000 1 1 1 1

import os
from data_processing import base_culane
root = os.path.join(base_culane, 'lists')
old_file_names = ['train_gt.txt', 'val_gt.txt', 'val', 'test.txt']
new_file_names = ['train.txt', 'valfast.txt', 'val', 'test.txt']
for i in range(len(old_file_names)):
    file_name = os.path.join(root, old_file_names[i])
    with open(file_name, 'r') as f:
        temp = f.readlines()
    for x in range(len(temp)):
        if 'test.txt' or 'val.txt' in file_name:
            temp[x] = temp[x].replace('.jpg', '')[1:]
        else:
            temp[x] = temp[x][1: temp[x].find('.jpg')] + temp[x][temp[x].find('.png') + 4:]
    file_name = os.path.join(root, new_file_names[i])
    with open(file_name, 'w') as f:
        f.writelines(temp)
