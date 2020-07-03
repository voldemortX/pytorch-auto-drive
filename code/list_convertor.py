# /clips/0313-1/6040/20.jpg /segGT6/0313-1/6040/20.png 0 1 1 1 1 0 =>
# 0313-1/6040/20 0 1 1 1 1 0

import os
root = '../../../dataset/tusimple/lists'
file_names = ['train.txt', 'val.txt', 'trainval.txt']
for file_name in file_names:
    file_name = os.path.join(root, file_name)
    with open(file_name, 'r') as f:
        temp = f.readlines()
    for x in range(len(temp)):
        temp[x] = temp[x][temp[x].find('clips/') + 6: temp[x].find('.jpg')] + temp[x][temp[x].find('.png') + 4:]
    with open(file_name, 'w') as f:
        f.writelines(temp)
