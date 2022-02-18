import os
import json
import argparse
import numpy as np
from tqdm import tqdm
from loader import SimpleKPLoader
from _utils import root_map, size_map
from importmagician import import_from
with import_from('./'):
    from utils.curve_utils import BezierCurve as Bezier, Polynomial as Poly
    from utils.common import warnings


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PytorchAutoDrive curve utils')
    parser.add_argument('--dataset', type=str, default='culane')
    parser.add_argument('--state', type=int, default=1, help='1: test set/2: val test')
    parser.add_argument('--fit-function', type=str, default='bezier', help='bezier/poly')
    parser.add_argument('--num-points', type=int, default=50, help='the number of sampled points')
    parser.add_argument('--order', type=int, default=2)
    parser.add_argument('--norm', action='store_true', default=False, help='normalize coordinates')
    args = parser.parse_args()

    root = root_map[args.dataset]
    image_size = size_map[args.dataset]
    bezier_threshold = 5
    image_set = 'test' if args.state == 1 else 'val'
    if args.dataset == 'llamas' and image_set != 'val':
        warnings.warn('LLAMAS test labels not available! Switching to validation set!')
        image_set = 'val'
    order = args.order
    lkp = SimpleKPLoader(root=root, image_set=image_set, data_set=args.dataset, image_size=image_size, norm=args.norm)
    lane_interpolate = True if args.dataset == 'curvelanes' else False
    keypoints = lkp.load_annotations()
    all_lanes = []
    for kps in tqdm(keypoints.keys()):
        coordinates = []
        for kp in keypoints[kps]:

            if args.fit_function == 'bezier':
                fcns = Bezier(order=order)
                fcns.get_control_points(kp[:, 0], kp[:, 1], interpolate=lane_interpolate)
            elif args.fit_function == 'poly':
                fcns = Poly(order=order)
                fcns.poly_fit(kp[:, 0], kp[:, 1], interpolate=lane_interpolate)
            else:
                raise ValueError

            if args.dataset == 'tusimple':

                temp = []
                if args.fit_function == 'bezier':
                    h_samples = [(160 + y * 10) for y in range(56)]
                    sampled_points = fcns.get_sample_point(n=args.num_points,
                                                           image_size=image_size if args.norm else None)
                    for h_sample in h_samples:
                        dis = np.abs(h_sample - sampled_points[:, 1])
                        idx = np.argmin(dis)
                        if dis[idx] > bezier_threshold:
                            temp.append(-2)
                        else:
                            temp.append(round(sampled_points[:, 0][idx], 3))
                    coordinates.append(temp)
                elif args.fit_function == 'poly':
                    if args.norm:
                        h_samples = [(160 + y * 10) / image_size[0] for y in range(56)]
                    else:
                        h_samples = [(160 + y * 10) for y in range(56)]
                    # sampled_points = fcns.get_sample_point(kp[:, 1])
                    start_y = kp[:, 1][0]
                    end_y = kp[:, 1][-1]
                    for h_sample in h_samples:
                        if h_sample < start_y:
                            temp.append(-2)
                        elif h_sample >= start_y and h_sample <= end_y:
                            temp.append(
                                round(fcns.compute_x_based_y(h_sample, image_size=image_size if args.norm else None),
                                      3))
                        elif h_sample > end_y:
                            temp.append(-2)
                    coordinates.append(temp)
                else:
                    raise ValueError
            else:
                if args.fit_function == 'bezier':
                    coordinates.append(
                        fcns.get_sample_point(n=args.num_points, image_size=image_size if args.norm else None))
                elif args.fit_function == 'poly':
                    coordinates.append(fcns.get_sample_point(kp[:, 1], image_size=image_size if args.norm else None))
                else:
                    raise ValueError

        # save the result
        if args.dataset in ['culane', 'llamas', 'curvelanes']:

            filepath = os.path.join('./output', kps)
            if args.dataset == 'llamas':
                filepath = filepath.replace('/color_images', '')
                filepath = filepath.replace('_color_rect', '')

            dir_name = filepath[:filepath.rfind('/')]
            if not os.path.exists(dir_name):
                os.makedirs(dir_name)
            with open(filepath, "w") as f:
                for coordinate in coordinates:
                    for j in range(len(coordinate)):
                        print("{} {}".format(round(coordinate[j][0], 3), round(coordinate[j][1]), 3), end=" ", file=f)
                    print(file=f)

        elif args.dataset == 'tusimple':
            formatted = {
                "h_samples": [160 + y * 10 for y in range(56)],
                "lanes": coordinates,
                "run_time": 0,
                "raw_file": kps
            }
            all_lanes.append(json.dumps(formatted))

    if args.dataset == 'tusimple':
        with open('./output/upperbound_' + args.fit_function + '_' + str(args.order) + '.json', 'w') as f:
            for lane in all_lanes:
                print(lane, end="\n", file=f)
