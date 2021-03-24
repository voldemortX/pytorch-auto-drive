from transforms import ToTensor, Normalize, RandomHorizontalFlip, Resize, RandomCrop, RandomTranslation, \
    ZeroPad, LabelMap, RandomScale, Compose
from utils.datasets import StandardSegmentationDataset
import torch


def init(input_sizes, std, mean, dataset, train_base, train_label_id_map,
         test_base=None, test_label_id_map=None, city_aug=0, workers=0):

    if test_base is None:
        test_base = train_base
    if test_label_id_map is None:
        test_label_id_map = train_label_id_map

    if dataset == 'voc':
        transform_test = Compose(
            [ToTensor(),
             ZeroPad(size=input_sizes),
             Normalize(mean=mean, std=std)])
    elif dataset == 'city' or dataset == 'gtav' or dataset == 'synthia':  # All the same size
        if city_aug == 2:  # ERFNet and ENet
            transform_test = Compose(
                [ToTensor(),
                 Resize(size_image=input_sizes, size_label=input_sizes),
                 LabelMap(test_label_id_map)])
        elif city_aug == 1:  # City big
            transform_test = Compose(
                [ToTensor(),
                 Resize(size_image=input_sizes, size_label=input_sizes),
                 Normalize(mean=mean, std=std),
                 LabelMap(test_label_id_map)])
    else:
        raise ValueError

    # Not the actual test set (i.e. validation set)
    test_set = StandardSegmentationDataset(root=test_base, image_set='val', transforms=transform_test,
                                           data_set='city' if dataset == 'gtav' or dataset == 'synthia' else dataset)

    val_loader = torch.utils.data.DataLoader(dataset=test_set, batch_size=1, num_workers=workers,
                                             shuffle=False)

    # Testing
    return val_loader
