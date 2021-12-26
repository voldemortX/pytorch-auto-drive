import os

from importmagician import import_from
with import_from('./'):
    from configs.semantic_segmentation.common.datasets._utils import CITYSCAPES_ROOT as base


def traverse(images_dir, data_list):
    for city in sorted(os.listdir(images_dir)):
        city_path = os.path.join(images_dir, city)
        for image in sorted(os.listdir(city_path)):
            temp = city + '/' + image.split('_leftImg8bit')[0] + '\n'
            data_list.append(temp)


# Traverse images
train_list = []
val_list = []
test_list = []
traverse(os.path.join(base, "leftImg8bit/train"), train_list)
traverse(os.path.join(base, "leftImg8bit/val"), val_list)
traverse(os.path.join(base, "leftImg8bit/test"), test_list)
print('Whole training set size: ' + str(len(train_list)))
print('Whole validation set size: ' + str(len(val_list)))
print('Whole test set size: ' + str(len(test_list)))

# Save training list
lists_dir = os.path.join(base, "data_lists")
if not os.path.exists(lists_dir):
    os.makedirs(lists_dir)
with open(os.path.join(lists_dir, "train.txt"), "w") as f:
    f.writelines(train_list)
with open(os.path.join(lists_dir, "val.txt"), "w") as f:
    f.writelines(val_list)
with open(os.path.join(lists_dir, "test.txt"), "w") as f:
    f.writelines(test_list)
print("Complete.")
