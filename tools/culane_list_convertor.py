# CULane (official lists)
# /driver_23_30frame/05151649_0422.MP4/00000.jpg /laneseg_label_w16/driver_23_30frame/05151649_0422.MP4/00000.png 1 1 1 1 =>
# /driver_23_30frame/05151649_0422.MP4/00000 1 1 1 1

import os
import yaml


with open('configs.yaml', 'r') as f:  # Safer and cleaner than box/EasyDict
    configs = yaml.load(f, Loader=yaml.Loader)
base = configs['CULANE']['BASE_DIR']
root = os.path.join(base, 'lists')
old_file_names = ['train_gt.txt', 'val_gt.txt', 'val.txt', 'test.txt']
new_file_names = ['train.txt', 'valfast.txt', 'val.txt', 'test.txt']
for i in range(len(old_file_names)):
    file_name = os.path.join(root, old_file_names[i])
    with open(file_name, 'r') as f:
        temp = f.readlines()
    for x in range(len(temp)):
        if new_file_names[i] == 'test.txt' or new_file_names[i] == 'val.txt':
            temp[x] = temp[x].replace('.jpg', '')[1:]
        else:
            temp[x] = temp[x][1: temp[x].find('.jpg')] + temp[x][temp[x].find('.png') + 4:]
    file_name = os.path.join(root, new_file_names[i])
    with open(file_name, 'w') as f:
        f.writelines(temp)
