import os
import yaml


# Pad with 0
def pad(x):
    zero = '0'
    length = len(x)
    if length < 5:
        x = zero * (5 - length) + x
    x += '\n'

    return x


# Count
start = 1
end = 24966
train_list = [pad(str(x)) for x in range(start, end + 1)]
print('Whole training set size: ' + str(len(train_list)))

# Save training list
with open('configs.yaml', 'r') as f:  # Safer and cleaner than box/EasyDict
    configs = yaml.load(f, Loader=yaml.Loader)
base = configs['GTAV']['BASE_DIR']
lists_dir = os.path.join(base, "data_lists")
if not os.path.exists(lists_dir):
    os.makedirs(lists_dir)
with open(os.path.join(lists_dir, "train.txt"), "w") as f:
    f.writelines(train_list)
print("Complete.")
