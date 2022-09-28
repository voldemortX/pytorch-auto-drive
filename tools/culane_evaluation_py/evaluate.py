# modified from Turoad/lanedet to align with original Cpp scripts
import os
import argparse
from functools import partial
from p_tqdm import p_map

from culane_metric import culane_metric


def load_culane_data_one(path):
    with open(path, 'r') as data_file:
        t = data_file.readlines()
    t = [line.split() for line in t]
    t = [list(map(float, lane)) for lane in t]
    t = [[(lane[i], lane[i + 1]) for i in range(0, len(lane), 2)] for lane in t]
    t = [lane for lane in t if len(lane) >= 2]

    return t


def load_culane_data(data_dir, file_list_path):
    with open(file_list_path, 'r') as file_list:
        filepaths = [
            os.path.join(data_dir, line[1 if line[0] == '/' else 0:].rstrip().replace('.jpg', '.lines.txt'))
            for line in file_list.readlines()
        ]

    data = []
    for path in filepaths:
        img_data = load_culane_data_one(path)
        data.append(img_data)

    return data


def eval_predictions(args, official=True):
    print('List file: {}'.format(args.list_path))
    print('Width lane: {}'.format(args.width))
    print('IoU threshold: {}'.format(args.threshold))
    print('Image height: {}'.format(args.img_height))
    print('Image width: {}'.format(args.img_width))
    print('Loading prediction data...')
    predictions = load_culane_data(args.pred_dir, args.list_path)
    print('Loading annotation data...')
    annotations = load_culane_data(args.anno_dir, args.list_path)
    print('Calculating metric in parallel...')
    img_shape = (args.img_height, args.img_width, 3)
    results = p_map(partial(culane_metric, width=args.width, iou_threshold=args.threshold, official=official, img_shape=img_shape),
                    predictions, annotations)
    total_tp = sum(tp for tp, _, _, _, _ in results)
    total_fp = sum(fp for _, fp, _, _, _ in results)
    total_fn = sum(fn for _, _, fn, _, _ in results)
    if total_tp == 0:
        precision = 0.0
        recall = 0.0
        f1 = 0.0
    else:
        precision = float(total_tp) / (total_tp + total_fp)
        recall = float(total_tp) / (total_tp + total_fn)
        f1 = 2 * precision * recall / (precision + recall)

    return {'TP': total_tp, 'FP': total_fp, 'FN': total_fn, 'Precision': precision, 'Recall': recall, 'F1': f1}


if __name__ == '__main__':
    # Settings
    # retain original defaults
    # remove unused/not needed: -i -f -s
    parser = argparse.ArgumentParser(description='CULane test')
    parser.add_argument('-a', '--anno-dir', type=str, help='directory for annotation files', default='/data/driving/eval_data/anno_label/')
    parser.add_argument('-d', '--pred-dir', type=str, help='directory for detection files', default='/data/driving/eval_data/predict_label/')
    parser.add_argument('-l', '--list-path', type=str, help='directory for image files', default='/data/driving/eval_data/img/')
    parser.add_argument('-w', '--width', type=int, help='width of the lanes', default=10)
    parser.add_argument('-t', '--threshold', type=float, help='threshold of iou', default=0.4)
    parser.add_argument('-c', '--img-width', type=int, help='cols (max image width)', default=1920)
    parser.add_argument('-r', '--img-height', type=int, help='rows (max image height)', default=1080)
    parser.add_argument('-o', '--output-path', type=str, help='result txt output path', default='./output.txt')

    _args = parser.parse_args()
    res = eval_predictions(_args)
    for k, v in res.items():
        print('{}: {:.6f}'.format(k, v) if type(v) == float else '{}: {}'.format(k, v))
    with open(_args.output_path, 'w') as f:
        f.write('file: {}\n'.format(_args.output_path))
        f.write('tp: {} fp: {} fn: {}\n'.format(res['TP'], res['FP'], res['FN']))
        f.write('precision: {:.6f}\n'.format(res['Precision']))
        f.write('recall: {:.6f}\n'.format(res['Recall']))
        f.write('Fmeasure: {:.6f}\n\n'.format(res['F1']))
