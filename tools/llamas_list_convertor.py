import os
from tqdm import tqdm
from llamas_evaluation.llamas_official_scripts import get_horizontal_values_for_four_lanes

from importmagician import import_from
with import_from('./'):
    from configs.lane_detection.common.datasets._utils import LLAMAS_ROOT as base
LLAMAS_H = 717

#
list_path = os.path.join(base, 'lists')
image_path = os.path.join(base, 'color_images')
label_path = os.path.join(base, 'labels')

if os.path.exists(list_path) is False:
    os.makedirs(list_path)
file_names = ['train', 'val', 'valfast', 'test']


def get_file_paths(dir, type):
    paths = []
    for root, dirs, files in os.walk(dir):
        for file in files:
            if file.endswith(type):
                paths.append(os.path.join(root, file))
    return paths


def coords2str(lane):
    s = ""
    for coords in lane:
        s = s + str(round(coords[0], 3)) + " "
        s = s + str(coords[1]) + " "
    s = s + '\n'
    return s


def get_txtfile(filepath, lanes):
    with open(filepath, 'a') as f:
        for lane in lanes:
            f.writelines(coords2str(lane))
    return 0


def existence2str(exist):
    s = ""
    for idx in exist:
        s = s + str(idx) + " "
    return s


def spline_annotation(json_name, image_name, get_txt):
    spline_lanes = get_horizontal_values_for_four_lanes(json_name)
    lanes = [[(x, y) for x, y in zip(lane, range(LLAMAS_H)) if x >= 0] for lane in spline_lanes]
    lanes_exist = [1 if len(lane) > 0 else 0 for lane in lanes]
    lanes = [lane for lane in lanes if len(lane) > 0]
    if get_txt is True:
        txt_path = image_name.replace('.png', '.lines.txt')
        get_txtfile(txt_path, lanes)
    return lanes_exist


def get_spline(filetype, filename, get_txt=False, existence=False, ant_exist=True):
    images_list = get_file_paths(os.path.join(image_path, filetype), ".png")
    images_list.sort()
    json_list = get_file_paths(os.path.join(label_path, filetype), ".json")
    if len(json_list) != 0:
        json_list.sort()
    length_of_list = len(images_list)
    for idx in tqdm(range(0, length_of_list)):
        lanes_exist = []
        if ant_exist:
            lanes_exist = spline_annotation(json_list[idx], images_list[idx], get_txt)
        with open(os.path.join(list_path, filename), 'a') as f:
            if existence is True:
                f.writelines(
                    images_list[idx][len(image_path) + 1:].replace('.png', '') + " " + existence2str(lanes_exist) + "\n")
            else:
                f.writelines(images_list[idx][len(image_path) + 1:].replace('.png', '') + "\n")
    return 0


def generate_spline_annotation():
    for file_name in file_names:
        if file_name == 'train':
            print(file_name+".txt is processing...")
            get_spline(file_name, file_name + '.txt', get_txt=True, existence=True, ant_exist=True)
        elif file_name == 'valfast':
            print(file_name + ".txt is processing...")
            get_spline('valid', file_name + '.txt', get_txt=True, existence=True, ant_exist=True)
        elif file_name == 'val':
            print(file_name + ".txt is processing...")
            get_spline('valid', file_name + '.txt', get_txt=False, existence=False, ant_exist=False)
        elif file_name == 'test':
            print(file_name + ".txt is processing...")
            get_spline(file_name, file_name + '.txt', get_txt=False, existence=False, ant_exist=False)

    return 0


if __name__ == '__main__':
    generate_spline_annotation()



