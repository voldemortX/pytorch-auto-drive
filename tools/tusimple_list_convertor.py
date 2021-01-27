# TuSimple (SAD lists)
# /clips/0313-1/6040/20.jpg /segGT6/0313-1/6040/20.png 1 1 1 1 1 1 =>
# 0313-1/6040/20 1 1 1 1 1 1
import os
import yaml


with open('configs.yaml', 'r') as f:  # Safer and cleaner than box/EasyDict
    configs = yaml.load(f, Loader=yaml.Loader)
base = configs['TUSIMPLE']['BASE_DIR']
root = os.path.join(base, 'lists')
old_file_names = ['list6_train.txt', 'list6_val.txt', 'list6_val.txt', 'list_test.txt']  # 6 lanes (actually <=5)
# old_file_names = ['list_train.txt', 'list_val.txt', 'list_val.txt', 'list_test.txt']  # 4 lanes
new_file_names = ['train.txt', 'valfast.txt', 'val.txt', 'test.txt']
for i in range(len(old_file_names)):
    file_name = os.path.join(root, old_file_names[i])
    with open(file_name, 'r') as f:
        temp = f.readlines()
    for x in range(len(temp)):
        if new_file_names[i] == 'test.txt' or new_file_names[i] == 'val.txt':
            temp[x] = temp[x][temp[x].find('clips/') + 6: temp[x].find('.jpg')] + '\n'
        else:
            temp[x] = temp[x][temp[x].find('clips/') + 6: temp[x].find('.jpg')] + temp[x][temp[x].find('.png') + 4:]
    file_name = os.path.join(root, new_file_names[i])
    with open(file_name, 'w') as f:
        f.writelines(temp)
