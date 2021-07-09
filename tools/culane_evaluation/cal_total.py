# Accumulate and calculate whole F1 score on CULane

import argparse
import fcntl

if __name__ == '__main__':
    # Settings
    parser = argparse.ArgumentParser(description='PyTorch 1.6.0')
    parser.add_argument('--exp-name', type=str, default='',
                        help='Name of experiment')
    args = parser.parse_args()

    filename = 'output/' + args.exp_name + '_iou0.5_split.txt'
    with open(filename, 'r') as f:
        temp = f.readlines()

    # Count
    tp = 0
    fp = 0
    fn = 0
    for i in range(9):
        line = temp[i * 6 + 1].replace('\n', '').split(' ')
        tp += int(line[1])
        fp += int(line[3])
        fn += int(line[5])

    # Calculate
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2 * precision * recall / (precision + recall) * 100

    # Log
    print('F1 score: ' + str(f1))
    with open('../../log.txt', 'a') as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        f.write(args.exp_name + ': ' + str(f1) + '\n')
        fcntl.flock(f, fcntl.LOCK_UN)
