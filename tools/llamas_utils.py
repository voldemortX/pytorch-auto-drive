# All following lines were taken from: https://github.com/karstenBehrendt/unsupervised_llamas
# Its license is copied here

# ##### Begin License ######
# MIT License

# Copyright (c) 2019 Karsten Behrendt, Robert Bosch LLC

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# ##### End License ######

# Start code under the previous license
import json
import os
import numpy as np
import tqdm


def _extend_lane(lane, projection_matrix):
    """Extends marker closest to the camera

    Adds an extra marker that reaches the end of the image

    Parameters
    ----------
    lane : iterable of markers
    projection_matrix : 3x3 projection matrix
    """
    # Unfortunately, we did not store markers beyond the image plane. That hurts us now
    # z is the orthongal distance to the car. It's good enough

    # The markers are automatically detected, mapped, and labeled. There exist faulty ones,
    # e.g., horizontal markers which need to be filtered
    filtered_markers = filter(
        lambda x: (x['pixel_start']['y'] != x['pixel_end']['y'] and x['pixel_start']['x'] != x['pixel_end']['x']),
        lane['markers'])
    # might be the first marker in the list but not guaranteed
    closest_marker = min(filtered_markers, key=lambda x: x['world_start']['z'])

    if closest_marker['world_start']['z'] < 0:  # This one likely equals "if False"
        return lane

    # World marker extension approximation
    x_gradient = (closest_marker['world_end']['x'] - closest_marker['world_start']['x']) / \
                 (closest_marker['world_end']['z'] - closest_marker['world_start']['z'])
    y_gradient = (closest_marker['world_end']['y'] - closest_marker['world_start']['y']) / \
                 (closest_marker['world_end']['z'] - closest_marker['world_start']['z'])

    zero_x = closest_marker['world_start']['x'] - (closest_marker['world_start']['z'] - 1) * x_gradient
    zero_y = closest_marker['world_start']['y'] - (closest_marker['world_start']['z'] - 1) * y_gradient

    # Pixel marker extension approximation
    pixel_x_gradient = (closest_marker['pixel_end']['x'] - closest_marker['pixel_start']['x']) / \
                       (closest_marker['pixel_end']['y'] - closest_marker['pixel_start']['y'])
    pixel_y_gradient = (closest_marker['pixel_end']['y'] - closest_marker['pixel_start']['y']) / \
                       (closest_marker['pixel_end']['x'] - closest_marker['pixel_start']['x'])

    pixel_zero_x = closest_marker['pixel_start']['x'] + (716 - closest_marker['pixel_start']['y']) * pixel_x_gradient
    if pixel_zero_x < 0:
        left_y = closest_marker['pixel_start']['y'] - closest_marker['pixel_start']['x'] * pixel_y_gradient
        new_pixel_point = (0, left_y)
    elif pixel_zero_x > 1276:
        right_y = closest_marker['pixel_start']['y'] + (1276 - closest_marker['pixel_start']['x']) * pixel_y_gradient
        new_pixel_point = (1276, right_y)
    else:
        new_pixel_point = (pixel_zero_x, 716)

    new_marker = {
        'lane_marker_id': 'FAKE',
        'world_end': {
            'x': closest_marker['world_start']['x'],
            'y': closest_marker['world_start']['y'],
            'z': closest_marker['world_start']['z']
        },
        'world_start': {
            'x': zero_x,
            'y': zero_y,
            'z': 1
        },
        'pixel_end': {
            'x': closest_marker['pixel_start']['x'],
            'y': closest_marker['pixel_start']['y']
        },
        'pixel_start': {
            'x': ir(new_pixel_point[0]),
            'y': ir(new_pixel_point[1])
        }
    }
    lane['markers'].insert(0, new_marker)

    return lane


class SplineCreator():
    """
    For each lane divder
      - all lines are projected
      - linearly interpolated to limit oscillations
      - interpolated by a spline
      - subsampled to receive individual pixel values

    The spline creation can be optimized!
      - Better spline parameters
      - Extend lowest marker to reach bottom of image would also help
      - Extending last marker may in some cases be interesting too
    Any help is welcome.

    Call create_all_points and get the points in self.sampled_points
    It has an x coordinate for each value for each lane

    """

    def __init__(self, json_path):
        self.json_path = json_path
        self.json_content = read_json(json_path)
        self.lanes = self.json_content['lanes']
        self.lane_marker_points = {}
        self.sampled_points = {}  # <--- the interesting part
        self.debug_image = np.zeros((717, 1276, 3), dtype=np.uint8)

    def _sample_points(self, lane, ypp=5, between_markers=True):
        """ Markers are given by start and endpoint. This one adds extra points
        which need to be considered for the interpolation. Otherwise the spline
        could arbitrarily oscillate between start and end of the individual markers

        Parameters
        ----------
        lane: polyline, in theory but there are artifacts which lead to inconsistencies
              in ordering. There may be parallel lines. The lines may be dashed. It's messy.
        ypp: y-pixels per point, e.g. 10 leads to a point every ten pixels
        between_markers : bool, interpolates inbetween dashes

        Notes
        -----
        Especially, adding points in the lower parts of the image (high y-values) because
        the start and end points are too sparse.
        Removing upper lane markers that have starting and end points mapped into the same pixel.
        """

        # Collect all x values from all markers along a given line. There may be multiple
        # intersecting markers, i.e., multiple entries for some y values
        x_values = [[] for i in range(717)]
        for marker in lane['markers']:
            x_values[marker['pixel_start']['y']].append(marker['pixel_start']['x'])

            height = marker['pixel_start']['y'] - marker['pixel_end']['y']
            if height > 2:
                slope = (marker['pixel_end']['x'] - marker['pixel_start']['x']) / height
                step_size = (marker['pixel_start']['y'] - marker['pixel_end']['y']) / float(height)
                for i in range(height + 1):
                    x = marker['pixel_start']['x'] + slope * step_size * i
                    y = marker['pixel_start']['y'] - step_size * i
                    x_values[ir(y)].append(ir(x))

        # Calculate average x values for each y value
        for y, xs in enumerate(x_values):
            if not xs:
                x_values[y] = -1
            else:
                x_values[y] = sum(xs) / float(len(xs))

        # In the following, we will only interpolate between markers if needed
        if not between_markers:
            return x_values  # TODO ypp

        # # interpolate between markers
        current_y = 0
        while x_values[current_y] == -1:  # skip missing first entries
            current_y += 1

        # Also possible using numpy.interp when accounting for beginning and end
        next_set_y = 0
        try:
            while current_y < 717:
                if x_values[current_y] != -1:  # set. Nothing to be done
                    current_y += 1
                    continue

                # Finds target x value for interpolation
                while next_set_y <= current_y or x_values[next_set_y] == -1:
                    next_set_y += 1
                    if next_set_y >= 717:
                        raise StopIteration

                x_values[current_y] = x_values[current_y - 1] + (x_values[next_set_y] - x_values[current_y - 1]) / \
                                      (next_set_y - current_y + 1)
                current_y += 1

        except StopIteration:
            pass  # Done with lane

        return x_values

    def _lane_points_fit(self, lane):
        # TODO name and docstring
        """ Fits spline in image space for the markers of a single lane (side)

        Parameters
        ----------
        lane: dict as specified in label

        Returns
        -------
        Pixel level values for curve along the y-axis

        Notes
        -----
        This one can be drastically improved. Probably fairly easy as well.
        """
        # NOTE all variable names represent image coordinates, interpolation coordinates are swapped!
        lane = _extend_lane(lane, self.json_content['projection_matrix'])
        sampled_points = self._sample_points(lane, ypp=1)
        self.sampled_points[lane['lane_id']] = sampled_points

        return sampled_points

    def create_all_points(self, ):
        """ Creates splines for given label """
        for lane in self.lanes:
            self._lane_points_fit(lane)


def get_horizontal_values_for_four_lanes(json_path):
    """ Gets an x value for every y coordinate for l1, l0, r0, r1

    This allows to easily train a direct curve approximation. For each value along
    the y-axis, the respective x-values can be compared, e.g. squared distance.
    Missing values are filled with -1. Missing values are values missing from the spline.
    There is no extrapolation to the image start/end (yet).
    But values are interpolated between markers. Space between dashed markers is not missing.

    Parameters
    ----------
    json_path: str
               path to label-file

    Returns
    -------
    List of [l1, l0, r0, r1], each of which represents a list of ints the length of
    the number of vertical pixels of the image

    Notes
    -----
    The points are currently based on the splines. The splines are interpolated based on the
    segmentation values. The spline interpolation has lots of room for improvement, e.g.
    the lines could be interpolated in 3D, a better approach to spline interpolation could
    be used, there is barely any error checking, sometimes the splines oscillate too much.
    This was used for a quick poly-line regression training only.
    """

    sc = SplineCreator(json_path)
    sc.create_all_points()

    l1 = sc.sampled_points.get('l1', [-1] * 717)
    l0 = sc.sampled_points.get('l0', [-1] * 717)
    r0 = sc.sampled_points.get('r0', [-1] * 717)
    r1 = sc.sampled_points.get('r1', [-1] * 717)

    lanes = [l1, l0, r0, r1]
    return lanes


def _filter_lanes_by_size(label, min_height=40):
    """ May need some tuning """
    filtered_lanes = []
    for lane in label['lanes']:
        lane_start = min([int(marker['pixel_start']['y']) for marker in lane['markers']])
        lane_end = max([int(marker['pixel_start']['y']) for marker in lane['markers']])
        if (lane_end - lane_start) < min_height:
            continue
        filtered_lanes.append(lane)
    label['lanes'] = filtered_lanes


def _filter_few_markers(label, min_markers=2):
    """Filter lines that consist of only few markers"""
    filtered_lanes = []
    for lane in label['lanes']:
        if len(lane['markers']) >= min_markers:
            filtered_lanes.append(lane)
    label['lanes'] = filtered_lanes


def _fix_lane_names(label):
    """ Given keys ['l3', 'l2', 'l0', 'r0', 'r2'] returns ['l2', 'l1', 'l0', 'r0', 'r1']"""

    # Create mapping
    l_counter = 0
    r_counter = 0
    mapping = {}
    lane_ids = [lane['lane_id'] for lane in label['lanes']]
    for key in sorted(lane_ids):
        if key[0] == 'l':
            mapping[key] = 'l' + str(l_counter)
            l_counter += 1
        if key[0] == 'r':
            mapping[key] = 'r' + str(r_counter)
            r_counter += 1
    for lane in label['lanes']:
        lane['lane_id'] = mapping[lane['lane_id']]


def read_json(json_path, min_lane_height=20):
    """ Reads and cleans label file information by path"""
    with open(json_path, 'r') as jf:
        label_content = json.load(jf)

    _filter_lanes_by_size(label_content, min_height=min_lane_height)
    _filter_few_markers(label_content, min_markers=2)
    _fix_lane_names(label_content)

    content = {'projection_matrix': label_content['projection_matrix'], 'lanes': label_content['lanes']}

    for lane in content['lanes']:
        for marker in lane['markers']:
            for pixel_key in marker['pixel_start'].keys():
                marker['pixel_start'][pixel_key] = int(marker['pixel_start'][pixel_key])
            for pixel_key in marker['pixel_end'].keys():
                marker['pixel_end'][pixel_key] = int(marker['pixel_end'][pixel_key])
            for pixel_key in marker['world_start'].keys():
                marker['world_start'][pixel_key] = float(marker['world_start'][pixel_key])
            for pixel_key in marker['world_end'].keys():
                marker['world_end'][pixel_key] = float(marker['world_end'][pixel_key])
    return content


def ir(some_value):
    """ Rounds and casts to int
    Useful for pixel values that cannot be floats
    Parameters
    ----------
    some_value : float
                 numeric value
    Returns
    --------
    Rounded integer
    Raises
    ------
    ValueError for non scalar types
    """
    return int(round(some_value))


# base = "/home/guoshaohua/dataset/llamas/"
base = "/home/guoshaohua/dataset/llamas/"
list_path = os.path.join(base, 'lists')
image_path = os.path.join(base, 'color_images')
label_path = os.path.join(base, 'labels')
LLAMAS_H = 717
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
    for idx in range(0, length_of_list):
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
            get_spline(file_name, file_name + '.txt', get_txt=True, existence=True, ant_exist=True)
            print(file_name+" completed...")
        elif file_name == 'valfast':
            get_spline('valid', file_name + '.txt', get_txt=True, existence=True, ant_exist=True)
            print(file_name + " completed...")
        elif file_name == 'val':
            get_spline('valid', file_name + '.txt', get_txt=False, existence=False, ant_exist=False)
            print(file_name + " completed...")
        elif file_name == 'test':
            get_spline(file_name, file_name + '.txt', get_txt=False, existence=False, ant_exist=False)
            print(file_name + " completed...")

    return 0



