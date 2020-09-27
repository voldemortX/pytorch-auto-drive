import os
from data_processing import base_gtav as base

# Count
start = 2500
end = 5000
train_list = ['0' + str(x) + '\n' for x in range(start, end + 1)]
print('Whole training set size: ' + str(len(train_list)))

# Save training list
lists_dir = os.path.join(base, "data_lists")
if not os.path.exists(lists_dir):
    os.makedirs(lists_dir)
with open(os.path.join(lists_dir, "train.txt"), "w") as f:
    f.writelines(train_list)
print("Complete.")
