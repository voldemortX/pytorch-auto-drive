# TuSimple (SAD lists)
# /clips/0313-1/6040/20.jpg /segGT6/0313-1/6040/20.png 0 1 1 1 1 0 =>
# 0313-1/6040/20 0 1 1 1 1 0

import os
root = '../../../dataset/tusimple/lists'
old_file_names = ['list6_train.txt', 'list6_val.txt', 'list6.txt', 'list_test.txt']
new_file_names = ['train.txt', 'val.txt', 'trainval.txt', 'test.txt']
for i in range(len(old_file_names)):
    file_name = os.path.join(root, old_file_names[i])
    with open(file_name, 'r') as f:
        temp = f.readlines()
    for x in range(len(temp)):
        if 'test.txt' in new_file_names[i]:
            temp[x] = temp[x][temp[x].find('clips/') + 6: temp[x].find('.jpg')] + '\n'
        else:
            temp[x] = temp[x][temp[x].find('clips/') + 6: temp[x].find('.jpg')] + temp[x][temp[x].find('.png') + 4:]
    file_name = os.path.join(root, new_file_names[i])
    with open(file_name, 'w') as f:
        f.writelines(temp)

# CULane (official lists)
# /driver_23_30frame/05151649_0422.MP4/00000.jpg /laneseg_label_w16/driver_23_30frame/05151649_0422.MP4/00000.png 1 1 1 1 =>
# /driver_23_30frame/05151649_0422.MP4/00000 1 1 1 1

root = '../../../dataset/culane/lists'
old_file_names = ['train_gt.txt', 'val_gt.txt', 'test.txt']
new_file_names = ['train.txt', 'val.txt', 'test.txt']
for i in range(len(old_file_names)):
    file_name = os.path.join(root, old_file_names[i])
    with open(file_name, 'r') as f:
        temp = f.readlines()
    for x in range(len(temp)):
        if 'test.txt' in file_name:
            temp[x] = temp[x].replace('.jpg', '')[1:]
        else:
            temp[x] = temp[x][1: temp[x].find('.jpg')] + temp[x][temp[x].find('.png') + 4:]
    file_name = os.path.join(root, new_file_names[i])
    with open(file_name, 'w') as f:
        f.writelines(temp)
