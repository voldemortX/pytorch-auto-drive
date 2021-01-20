from .segmentation import *
from .lane_as_segmentation import *
from .tusimple import *
from .culane import *

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

# Here 'training resize min' is also the final training crop size as RandomResize & RandomCrop are used together
# For PASCAL VOC 2012
base_voc = '../data/VOCtrainval_11-May-2012/VOCdevkit/VOC2012'
sizes_voc = [(321, 321), (505, 505), (505, 505)]  # training resize min/training resize max/testing label size
num_classes_voc = 21
colors_voc = [[0, 0, 0],
              [128, 0, 0], [0, 128, 0], [128, 128, 0], [0, 0, 128],
              [128, 0, 128], [0, 128, 128], [128, 128, 128], [64, 0, 0],
              [192, 0, 0], [64, 128, 0], [192, 128, 0], [64, 0, 128],
              [192, 0, 128], [64, 128, 128], [192, 128, 128], [0, 64, 0],
              [128, 64, 0], [0, 192, 0], [128, 192, 0], [0, 64, 128],
              [255, 255, 255]]
categories_voc = ['Background',
                  'Aeroplane', 'Bicycle', 'Bird', 'Boat',
                  'Bottle', 'Bus', 'Car', 'Cat',
                  'Chair', 'Cow', 'Diningtable', 'Dog',
                  'Horse', 'Motorbike', 'Person', 'Pottedplant',
                  'Sheep', 'Sofa', 'Train', 'Tvmonitor']

# For cityscapes (19 classes, ignore as black, no such thing as background)
base_city = '../../../dataset/cityscapes'
sizes_city = [(256, 512), (512, 1024), (512, 1024)]  # training resize min/training resize max/testing label size
sizes_city_big = [(512, 1024), (512, 1024), (1024, 2048)]  # training resize min/training resize max/testing label size
sizes_city_erfnet = [(512, 1024), (64, 128), (512, 1024)]  # input/encoder output/testing label size
weights_city_erfnet = [2.8149201869965, 6.9850029945374, 3.7890393733978, 9.9428062438965,
                       9.7702074050903, 9.5110931396484, 10.311357498169, 10.026463508606,
                       4.6323022842407, 9.5608062744141, 7.8698215484619, 9.5168733596802,
                       10.373730659485, 6.6616044044495, 10.260489463806, 10.287888526917,
                       10.289801597595, 10.405355453491, 10.138095855713]
num_classes_city = 19
colors_city = [
    [128, 64, 128], [244, 35, 232], [70, 70, 70], [102, 102, 156],
    [190, 153, 153], [153, 153, 153], [250, 170, 30], [220, 220, 0],
    [107, 142, 35], [152, 251, 152], [70, 130, 180], [220, 20, 60],
    [255, 0, 0], [0, 0, 142], [0, 0, 70], [0, 60, 100],
    [0, 80, 100], [0, 0, 230], [119, 11, 32],
    [0, 0, 0]]
categories_city = [
    'road', 'sidewalk', 'building', 'wall',
    'fence', 'pole', 'traffic light', 'traffic sign',
    'vegetation', 'terrain', 'sky', 'person',
    'rider', 'car', 'truck', 'bus',
    'train', 'motorcycle', 'bicycle']
label_id_map_city = [255, 255, 255, 255, 255, 255, 255,
                     0,   1,   255, 255, 2,   3,   4,
                     255, 255, 255, 5,   255, 6,   7,
                     8,   9,   10,  11,  12,  13,  14,
                     15,  255, 255, 16,  17,  18]
train_cities = ['aachen', 'bremen', 'darmstadt', 'erfurt', 'hanover',
                'krefeld', 'strasbourg', 'tubingen', 'weimar', 'bochum',
                'cologne', 'dusseldorf', 'hamburg', 'jena', 'monchengladbach',
                'stuttgart', 'ulm', 'zurich']

# For GTAV (19 classes, ignore as black, no such thing as background)
base_gtav = '../../../dataset/gtav'
sizes_gtav = [(512, 1024), (1054, 1912), (1024, 2048)]  # training crop size/max size/testing label size
sizes_gtav_erfnet = [(512, 1024), (64, 128), (512, 1024)]  # input/encoder output/testing label size

# For SYNTHIA (23 classes, ignore as black, no such thing as background, mapped to Cityscapes)
base_synthia = '../../../dataset/syn/SYNTHIA_RAND_CITYSCPAES'
sizes_synthia = [(512, 1024), (760, 1280), (1024, 2048)]  # training resize min/training resize max/testing label size
sizes_synthia_erfnet = [(512, 1024), (64, 128), (512, 1024)]  # input/encoder output/testing label size
label_id_map_synthia = [255, 10,  2,   0, 1,   4,
                        8,   5,   13,  7, 11,  18,
                        17,  255, 255, 6, 9,   12,
                        14,  15,  16,  3, 255]
iou_16 = [0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 15, 17, 18]
iou_13 = [0, 1, 2, 6, 7, 8, 10, 11, 12, 13, 15, 17, 18]

# For TuSimple
base_tusimple = '../../../dataset/tusimple'
sizes_tusimple = [(360, 640), (720, 1280)]  # training size/actual size
num_classes_tusimple = 5
weights_tusimple = [0.4, 1, 1, 1, 1]
gap_tusimple = 10  # Y pixel gap per sampling point
ppl_tusimple = 56  # Points per lane


# For CULane
base_culane = '../../../dataset/culane'
sizes_culane = [(288, 800), (590, 1640)]  # training size/actual size
num_classes_culane = 5
weights_culane = [0.4, 1, 1, 1, 1]
gap_culane = 20  # Y pixel gap per sampling point
ppl_culane = 18  # Points per lane
