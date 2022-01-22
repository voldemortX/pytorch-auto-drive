import argparse
import cv2
import numpy as np
try:
    import ujson as json
except ImportError:
    import json
from tqdm import tqdm
from tabulate import tabulate
from scipy.spatial import distance


def show_preds(pred, gt):
    img = np.zeros((720, 1280, 3), dtype=np.uint8)
    print(len(gt), 'gts and', len(pred), 'preds')
    for lane in gt:
        for p in lane:
            cv2.circle(img, tuple(map(int, p)), 5, thickness=-1, color=(255, 0, 255))
    for lane in pred:
        for p in lane:
            cv2.circle(img, tuple(map(int, p)), 4, thickness=-1, color=(0, 255, 0))
    cv2.imshow('img', img)
    cv2.waitKey(0)


def area_distance(pred_x, pred_y, gt_x, gt_y, placeholder=np.nan):
    pred = np.vstack([pred_x, pred_y]).T
    gt = np.vstack([gt_x, gt_y]).T

    # pred = pred[pred[:, 0] > 0][:3, :]
    # gt = gt[gt[:, 0] > 0][:5, :]

    dist_matrix = distance.cdist(pred, gt, metric='euclidean')

    dist = 0.5 * (np.min(dist_matrix, axis=0).sum() + np.min(dist_matrix, axis=1).sum())
    d = np.max(gt_y) - np.min(gt_y)
    if d == 0:
        d = 1.0
    dist /= d

    return dist


def area_metric(pred, gt, debug=None):
    pred = sorted(pred, key=lambda ps: abs(ps[0][0] - 720 / 2.))[:2]
    gt = sorted(gt, key=lambda ps: abs(ps[0][0] - 720 / 2.))[:2]
    if len(pred) == 0:
        return 0., 0., len(gt)
    line_dists = []
    fp = 0.
    matched = 0.
    gt_matches = [False] * len(gt)
    pred_matches = [False] * len(pred)
    pred_dists = [None] * len(pred)

    distances = np.ones((len(gt), len(pred)), dtype=np.float32)
    for i_gt, gt_points in enumerate(gt):
        x_gts = [x for x, _ in gt_points]
        y_gts = [y for _, y in gt_points]
        for i_pred, pred_points in enumerate(pred):
            x_preds = [x for x, _ in pred_points]
            y_preds = [y for _, y in pred_points]
            distances[i_gt, i_pred] = area_distance(x_preds, y_preds, x_gts, y_gts)

    best_preds = np.argmin(distances, axis=1)
    best_gts = np.argmin(distances, axis=0)
    fp = 0.
    fn = 0.
    dist = 0.
    is_fp = []
    is_fn = []
    for i_pred, best_gt in enumerate(best_gts):
        if best_preds[best_gt] == i_pred:
            dist += distances[best_gt, i_pred]
            is_fp.append(False)
        else:
            fp += 1
            is_fp.append(True)
    for i_gt, best_pred in enumerate(best_preds):
        if best_gts[best_pred] != i_gt:
            fn += 1
            is_fn.append(True)
        else:
            is_fn.append(False)
    if debug:
        print('is fp')
        print(is_fp)
        print('is fn')
        print(is_fn)
        print('distances')
        dists = np.min(distances, axis=0)
        dists[np.array(is_fp)] = 0
        print(dists)
        show_preds(pred, gt)

    return dist, fp, fn


def convert_tusimple_format(json_gt):
    output = []
    for data in json_gt:
        lanes = [[(x, y) for (x, y) in zip(lane, data['h_samples']) if x >= 0] for lane in data['lanes']
                 if any(x > 0 for x in lane)]
        output.append({
            'raw_file': data['raw_file'],
            'run_time': data['run_time'] if 'run_time' in data else None,
            'lanes': lanes
        })
    return output


def eval_json(pred_file, gt_file, json_type=None, debug=False):
    try:
        json_pred = [json.loads(line) for line in open(pred_file).readlines()]
    except BaseException as e:
        raise Exception('Fail to load json file of the prediction.')
    json_gt = [json.loads(line) for line in open(gt_file).readlines()]
    if len(json_gt) != len(json_pred):
        raise Exception('We do not get the predictions of all the test tasks')

    if json_type == 'tusimple':
        for gt, pred in zip(json_gt, json_pred):
            pred['h_samples'] = gt['h_samples']
        json_gt = convert_tusimple_format(json_gt)
        json_pred = convert_tusimple_format(json_pred)
    gts = {l['raw_file']: l for l in json_gt}

    total_distance, total_fp, total_fn, run_time = 0., 0., 0., 0.
    for pred in tqdm(json_pred):
        if 'raw_file' not in pred or 'lanes' not in pred:
            raise Exception('raw_file or lanes not in some predictions.')
        raw_file = pred['raw_file']
        pred_lanes = pred['lanes']
        run_time += pred['run_time'] if 'run_time' in pred else 1.

        if raw_file not in gts:
            raise Exception('Some raw_file from your predictions do not exist in the test tasks.')
        gt = gts[raw_file]
        gt_lanes = gt['lanes']

        distance, fp, fn = area_metric(pred_lanes, gt_lanes, debug=debug)

        total_distance += distance
        total_fp += fp
        total_fn += fn

    num = len(gts)
    return json.dumps([{
        'name': 'Distance',
        'value': total_distance / num,
        'order': 'desc'
    }, {
        'name': 'FP',
        'value': total_fp,
        'order': 'asc'
    }, {
        'name': 'FN',
        'value': total_fn,
        'order': 'asc'
    }
    ])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="PytorchAutoDrive curve utils")
    parser.add_argument('--preds', required=True, type=str, help=".json with the predictions")
    parser.add_argument('--gt', required=True, type=str, help=".json with the GT")
    parser.add_argument('--gt-type', type=str, help='pass `tusimple` if using the TuSimple file format')
    parser.add_argument('--debug', action='store_true', help='show metrics and preds/gts')
    argv = vars(parser.parse_args())

    result = json.loads(eval_json(argv['preds'], argv['gt'], argv['gt_type'], argv['debug']))

    # pretty-print
    table = {}
    for metric in result:
        if metric['name'] not in table.keys():
            table[metric['name']] = []
        table[metric['name']].append(metric['value'])
    print(tabulate(table, headers='keys'))
