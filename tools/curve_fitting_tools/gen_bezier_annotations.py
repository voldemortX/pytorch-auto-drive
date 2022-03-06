import argparse
import json
import os
from tqdm import tqdm
from loader import SimpleKPLoader
from _utils import root_map, size_map
from importmagician import import_from
with import_from('./'):
    from utils.curve_utils import BezierCurve as Bezier


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PytorchAutoDrive Bezier curve GT generation')
    parser.add_argument('--dataset', type=str, default='culane')
    parser.add_argument('--image-set', type=str, default='train', help='train.txt/test.txt/val.txt/valfast.txt')
    parser.add_argument('--order', type=int, default=3, help='the order of curve')
    parser.add_argument('--norm', action='store_true', default=False, help='normalize coordinates')
    args = parser.parse_args()

    root = root_map[args.dataset]
    image_size = size_map[args.dataset]
    order = args.order
    lane_interpolate = True if args.dataset == 'curvelanes' else False
    lkp = SimpleKPLoader(root=root, image_set=args.image_set, data_set=args.dataset, image_size=image_size, norm=False)
    keypoints = lkp.load_annotations()
    all_lanes_kps = []
    for kps in tqdm(keypoints.keys()):
        temp = []
        for kp in keypoints[kps]:
            if kp.shape[0] == 0:
                continue
            fcns = Bezier(order=order)
            fcns.get_control_points(kp[:, 0], kp[:, 1], interpolate=lane_interpolate)
            matrix = fcns.save_control_points()
            flatten = [round(p, 3) for sub_m in matrix for p in sub_m]
            temp.append(flatten)

        formatted = {
            "raw_file": kps,
            "bezier_control_points": temp
        }
        all_lanes_kps.append(json.dumps(formatted))

    dir_name = os.path.join(root, 'bezier_labels')
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    save_path = os.path.join(dir_name, args.image_set + "_" + str(args.order) + ".json")
    with open(save_path, 'w') as f:
        for lane in all_lanes_kps:
            print(lane, end="\n", file=f)
