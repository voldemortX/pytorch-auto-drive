import os
import numpy as np
from tqdm import tqdm
import json
from collections import OrderedDict


class SimpleKPLoader(object):
    def __init__(self, root, image_size, image_set='test', data_set='tusimple', norm=False):

        self.image_set = image_set
        self.data_set = data_set
        self.root = root
        self.norm = norm
        self.image_height = image_size[0]
        self.image_width = image_size[-1]

        if self.image_set == 'test' and data_set == 'llamas':
            raise ValueError

        if data_set == 'tusimple':
            self.image_dir = root
        elif data_set == 'culane':
            self.image_dir = root
            self.annotations_suffix = '.lines.txt'
        elif data_set == 'curvelanes':
            self.image_dir = root
            self.annotations_suffix = '.lines.txt'
        elif data_set == 'llamas':
            self.image_dir = os.path.join(root, 'color_images')
            self.annotations_suffix = '.lines.txt'
        else:
            raise ValueError

        self.splits_dir = os.path.join(root, 'lists')

    def load_txt_path(self, dataset):
        split_f = os.path.join(self.splits_dir, self.image_set + '.txt')
        with open(split_f, "r") as f:
            contents = [x.strip() for x in f.readlines()]
        if self.image_set in ['test', 'val']:
            path_lists = [os.path.join(self.image_dir, x + self.annotations_suffix) for x in contents]
        elif self.image_set == 'train':
            if dataset == 'curvelanes':
                path_lists = [os.path.join(self.image_dir, x + self.annotations_suffix) for x in contents]
            else:
                path_lists = [os.path.join(self.image_dir, x[:x.find(' ')] +
                                           self.annotations_suffix) for x in contents]
        else:
            raise ValueError

        return path_lists

    def load_json(self):
        if self.image_set == 'test':
            json_name = [os.path.join(self.image_dir, 'test_label.json')]
        elif self.image_set == 'val':
            json_name = [os.path.join(self.image_dir, 'label_data_0531.json')]
        elif self.image_set == 'train':
            json_name = [os.path.join(self.image_dir, 'label_data_0313.json'),
                            os.path.join(self.image_dir, 'label_data_0601.json')]
        else:
            raise ValueError

        return json_name

    def get_points_in_txtfile(self, file_path):
        coords = []
        with open(file_path, 'r') as f:
            for lane in f.readlines():
                lane = lane.split(' ')
                coord = []
                for i in range(0, len(lane) - 1, 2):
                    if float(lane[i]) >= 0:
                        coord.append([float(lane[i]), float(lane[i + 1])])
                coord = np.array(coord)
                if self.norm and len(coord) != 0:
                    coord[:, 0] = coord[:, 0] / self.image_width
                    coord[:, -1] = coord[:, -1] / self.image_height
                coords.append(coord)

        return coords

    def get_points_in_json(self, itm):
        lanes = itm['lanes']
        coords_list = []
        h_sample = itm['h_samples']
        for lane in lanes:
            coord = []
            for x, y in zip(lane, h_sample):
                if x >= 0:
                    coord.append([float(x), float(y)])
            coord = np.array(coord)
            if self.norm and len(coord) != 0:
                coord[:, 0] = coord[:, 0] / self.image_width
                coord[:, -1] = coord[:, -1] / self.image_height
            coords_list.append(coord)

        return coords_list

    def load_annotations(self):
        print('Loading dataset...')
        coords = OrderedDict()
        if self.data_set in ['culane', 'llamas', 'curvelanes']:
            file_lists = self.load_txt_path(dataset=self.data_set)
            for f in tqdm(file_lists):
                coords[f[len(self.root) + 1:]] = self.get_points_in_txtfile(f)
        elif self.data_set == 'tusimple':
            jsonfiles = self.load_json()

            results = []
            for jsonfile in jsonfiles:
                with open(jsonfile, 'r') as f:
                    results += [json.loads(x.strip()) for x in f.readlines()]
            for lane_json in tqdm(results):
                coords[lane_json['raw_file']] = self.get_points_in_json(lane_json)
        else:
            raise ValueError
        print('Finished')

        return coords

    def concat_jsons(self, filenames):
        # Concat tusimple lists in jsons (actually only each line is json)
        results = []
        for filename in filenames:
            with open(filename, 'r') as f:
                results += [json.loads(x.strip()) for x in f.readlines()]

        return results
