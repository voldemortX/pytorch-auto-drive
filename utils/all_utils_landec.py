import os
import cv2
import torch
import time
import ujson as json
import numpy as np
from tqdm import tqdm
if torch.__version__ >= '1.6.0':
    from torch.cuda.amp import autocast, GradScaler
else:
    from .torch_amp_dummy import autocast, GradScaler
from torchvision_models.segmentation import erfnet_resnet, deeplabv1_vgg16, deeplabv1_resnet18, deeplabv1_resnet34, \
    deeplabv1_resnet50, deeplabv1_resnet101, enet_
from torchvision_models.lane_detection import LSTR, RESANet
from torchvision_models.lane_detection.utils import lane_pruning
from utils.datasets import StandardLaneDetectionDataset, TuSimple, CULane, LLAMAS, dict_collate_fn
from transforms import ToTensor, Normalize, Resize, RandomRotation, RandomCrop, RandomHorizontalFlip, \
    RandomLighting, ColorJitter, RandomApply, Compose
from utils.all_utils_semseg import save_checkpoint, ConfusionMatrix
from .ddp_utils import reduce_dict, is_main_process


def erfnet_tusimple(num_classes, scnn=False, pretrained_weights='erfnet_encoder_pretrained.pth.tar'):
    # Define ERFNet for TuSimple (With only ImageNet pretraining)
    return erfnet_resnet(pretrained_weights=pretrained_weights, num_classes=num_classes, num_lanes=num_classes - 1,
                         dropout_1=0.3, dropout_2=0.3, flattened_size=4400, scnn=scnn)


def erfnet_culane(num_classes, scnn=False, pretrained_weights='erfnet_encoder_pretrained.pth.tar'):
    # Define ERFNet for CULane (With only ImageNet pretraining)
    return erfnet_resnet(pretrained_weights=pretrained_weights, num_classes=num_classes, num_lanes=num_classes - 1,
                         dropout_1=0.1, dropout_2=0.1, flattened_size=4500, scnn=scnn)


def erfnet_llamas(num_classes, scnn=False, pretrained_weights='erfnet_encoder_pretrained.pth.tar'):
    # Define ERFNet for CULane (With only ImageNet pretraining)
    return erfnet_resnet(pretrained_weights=pretrained_weights, num_classes=num_classes, num_lanes=num_classes - 1,
                         dropout_1=0.3, dropout_2=0.3, flattened_size=4400, scnn=scnn)


def vgg16_tusimple(num_classes, scnn=False, pretrained_weights='pytorch-pretrained'):
    # Define Vgg16 for Tusimple (With only ImageNet pretraining)
    return deeplabv1_vgg16(pretrained_weights=pretrained_weights, num_classes=num_classes, num_lanes=num_classes - 1,
                           dropout_1=0.1, flattened_size=6160, scnn=scnn)


def vgg16_llamas(num_classes, scnn=False, pretrained_weights='pytorch-pretrained'):
    # Define Vgg16 for Tusimple (With only ImageNet pretraining)
    return deeplabv1_vgg16(pretrained_weights=pretrained_weights, num_classes=num_classes, num_lanes=num_classes - 1,
                           dropout_1=0.1, flattened_size=4400, scnn=scnn)


def vgg16_culane(num_classes, scnn=False, pretrained_weights='pytorch-pretrained'):
    # Define Vgg16 for CULane (With only ImageNet pretraining)
    return deeplabv1_vgg16(pretrained_weights=pretrained_weights, num_classes=num_classes, num_lanes=num_classes - 1,
                           dropout_1=0.1, flattened_size=4500, scnn=scnn)


def resnet_tusimple(num_classes, backbone_name='resnet18', spatial_conv=None):
    # Define ResNets for Tusimple (With only ImageNet pretraining)
    if spatial_conv is None or spatial_conv == 'scnn':
        scnn = spatial_conv is not None
        model_map = {
            'resnet18': deeplabv1_resnet18,
            'resnet34': deeplabv1_resnet34,
            'resnet50': deeplabv1_resnet50,
            'resnet101': deeplabv1_resnet101,
        }
        return model_map[backbone_name](pretrained=False, num_classes=num_classes, num_lanes=num_classes - 1,
                                        channel_reduce=128, flattened_size=6160, scnn=scnn)
    elif spatial_conv == 'resa':
        return RESANet(num_classes=num_classes, backbone_name=backbone_name, flattened_size=4400, channel_reduce=128)
    else:
        raise ValueError('spatial_conv must be None, scnn or resa! Not {}'.format(spatial_conv))


def resnet_culane(num_classes, backbone_name='resnet18', spatial_conv=None):
    # Define ResNets for CULane (With only ImageNet pretraining)
    if spatial_conv is None or spatial_conv == 'scnn':
        scnn = spatial_conv is not None
        model_map = {
            'resnet18': deeplabv1_resnet18,
            'resnet34': deeplabv1_resnet34,
            'resnet50': deeplabv1_resnet50,
            'resnet101': deeplabv1_resnet101,
        }
        return model_map[backbone_name](pretrained=False, num_classes=num_classes, num_lanes=num_classes - 1,
                                        channel_reduce=128, flattened_size=4500, scnn=scnn)
    elif spatial_conv == 'resa':
        return RESANet(num_classes=num_classes, backbone_name=backbone_name, flattened_size=4500, channel_reduce=128)
    else:
        raise ValueError('spatial_conv must be None, scnn or resa! Not {}'.format(spatial_conv))


def resnet_llamas(num_classes, backbone_name='resnet18', spatial_conv=None):
    # Define ResNets for LLAMAS (With only ImageNet pretraining)
    if spatial_conv is None or spatial_conv == 'scnn':
        scnn = spatial_conv is not None
        model_map = {
            'resnet18': deeplabv1_resnet18,
            'resnet34': deeplabv1_resnet34,
            'resnet50': deeplabv1_resnet50,
            'resnet101': deeplabv1_resnet101,
        }
        return model_map[backbone_name](pretrained=False, num_classes=num_classes, num_lanes=num_classes - 1,
                                        channel_reduce=128, flattened_size=4400, scnn=scnn)
    elif spatial_conv == 'resa':
        return RESANet(num_classes=num_classes, backbone_name=backbone_name, flattened_size=4400, channel_reduce=128)
    else:
        raise ValueError('spatial_conv must be None, scnn or resa! Not {}'.format(spatial_conv))


def enet_tusimple(num_classes, encoder_only, continue_from):
    return enet_(num_classes=num_classes, num_lanes=num_classes - 1, dropout_1=0.01, dropout_2=0.1, flattened_size=4400,
                 encoder_only=encoder_only, pretrained_weights=continue_from if not encoder_only else None)


def enet_culane(num_classes, encoder_only, continue_from):
    return enet_(num_classes=num_classes, num_lanes=num_classes - 1, dropout_1=0.01, dropout_2=0.1, flattened_size=4500,
                 encoder_only=encoder_only, pretrained_weights=continue_from if not encoder_only else None)


def lstr_resnet(num_classes_max=7, backbone_name='resnet18s', expansion=1, aux_loss=True):
    # LSTR with ResNet backbone
    # The original LSTR-ResNet18 modified backbone is `resnet18s`,
    # where network width is controlled by expansion
    # aux_loss is only used in training
    if backbone_name in ['resnet18', 'resnet34']:
        expansion = 4
    elif backbone_name in ['resnet50', 'resnet101']:
        expansion = 16

    return LSTR(num_queries=num_classes_max, backbone_name=backbone_name, expansion=expansion, aux_loss=aux_loss)


def init(batch_size, state, input_sizes, dataset, mean, std, base, workers=10, method='baseline',
         aug_level=0, eigen_value=None, eigen_vector=None, ddp=False):
    # Return data_loaders
    # depending on whether the state is
    # 0: training
    # 1: fast validation by mean IoU (validation set)
    # 2: just testing (test set)
    # 3: just testing (validation set)
    # aug_level (implemented as a int instead of bool, for future extensions):
    # 0: Standard resize + small random rotation (3 degrees)
    # 1: PolyLaneNet/LSTR augmentation

    # Transformations
    # ! Can't use torchvision.Transforms.Compose
    transforms_test = Compose(
        [Resize(size_image=input_sizes[0], size_label=input_sizes[0]),
         ToTensor(),
         Normalize(mean=mean, std=std)])
    if aug_level == 0:
        transforms_train = Compose(
            [Resize(size_image=input_sizes[0], size_label=input_sizes[0]),
             RandomRotation(degrees=3),
             ToTensor(),
             Normalize(mean=mean, std=std, normalize_target=True if method == 'lstr' else False)])
    elif aug_level == 1:
        transforms_train = Compose(
            [RandomApply([
                RandomRotation(degrees=10),
                RandomHorizontalFlip(flip_prob=0.5),
                RandomCrop(size=(int(input_sizes[1][0] * 0.9), int(input_sizes[1][1] * 0.9)))
             ], apply_prob=10/11),
             Resize(size_image=input_sizes[0], size_label=input_sizes[0]),
             ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.2),
             ToTensor(),
             RandomLighting(mean=0.0, std=0.1, eigen_value=eigen_value, eigen_vector=eigen_vector),
             Normalize(mean=mean, std=std, normalize_target=True if method == 'lstr' else False)])
    else:
        raise ValueError

    # Batch builder
    if method == 'lstr':
        collate_fn = dict_collate_fn
    else:
        collate_fn = None

    if state == 0:
        if method == 'lstr':
            if dataset == 'tusimple':
                data_set = TuSimple(root=base, image_set='train', transforms=transforms_train,
                                    padding_mask=False, process_points=True)
            elif dataset == 'culane':
                data_set = CULane(root=base, image_set='train', transforms=transforms_train,
                                  padding_mask=False, process_points=True)
            elif dataset == 'llamas':
                data_set = LLAMAS(root=base, image_set='train', transforms=transforms_train,
                                  padding_mask=False, process_points=True)
            else:
                raise ValueError
        else:
            data_set = StandardLaneDetectionDataset(root=base, image_set='train', transforms=transforms_train,
                                                    data_set=dataset)
        validation_set = StandardLaneDetectionDataset(root=base, image_set='valfast',
                                                      transforms=transforms_test, data_set=dataset)
        # DDP
        if ddp:
            train_sampler = torch.utils.data.distributed.DistributedSampler(data_set)
        else:
            train_sampler = torch.utils.data.RandomSampler(data_set)
        data_loader = torch.utils.data.DataLoader(dataset=data_set, batch_size=batch_size, collate_fn=collate_fn,
                                                  sampler=train_sampler, num_workers=workers)
        validation_loader = torch.utils.data.DataLoader(dataset=validation_set, batch_size=batch_size * 4,
                                                        num_workers=workers, shuffle=False, collate_fn=collate_fn)

        return data_loader, validation_loader, train_sampler

    elif state == 1 or state == 2 or state == 3:
        image_sets = ['valfast', 'test', 'val']
        if method == 'lstr':
            if dataset == 'tusimple':
                data_set = TuSimple(root=base, image_set=image_sets[state - 1], transforms=transforms_test,
                                    padding_mask=False, process_points=False)
            elif dataset == 'culane':
                data_set = CULane(root=base, image_set=image_sets[state - 1], transforms=transforms_test,
                                  padding_mask=False, process_points=False)
            elif dataset == 'llamas':
                data_set = LLAMAS(root=base, image_set=image_sets[state - 1], transforms=transforms_test,
                                  padding_mask=False, process_points=False)

            else:
                raise ValueError
        else:
            data_set = StandardLaneDetectionDataset(root=base, image_set=image_sets[state - 1],
                                                    transforms=transforms_test, data_set=dataset)
        data_loader = torch.utils.data.DataLoader(dataset=data_set, batch_size=batch_size, collate_fn=collate_fn,
                                                  num_workers=workers, shuffle=False)
        return data_loader
    else:
        raise ValueError


def train_schedule(writer, loader, validation_loader, val_num_steps, device, criterion, net, optimizer, lr_scheduler,
                   num_epochs, is_mixed_precision, input_sizes, exp_name, num_classes, method='baseline',
                   ddp=False, train_sampler=None):
    # Should be the same as segmentation, given customized loss classes
    net.train()
    epoch = 0
    running_loss = None  # Dict logging for every loss (too many losses in this task)
    loss_num_steps = int(len(loader) / 10) if len(loader) > 10 else 1
    if is_mixed_precision:
        scaler = GradScaler()

    # Training
    best_validation = 0
    while epoch < num_epochs:
        net.train()
        if ddp:
            train_sampler.set_epoch(epoch)
        time_now = time.time()
        for i, data in enumerate(loader, 0):
            if method == 'lstr':
                inputs, labels = data
                inputs = inputs.to(device)
                labels = [{k: v.to(device) for k, v in label.items()} for label in labels]  # Seems slow
            else:
                inputs, labels, lane_existence = data
                inputs, labels, lane_existence = inputs.to(device), labels.to(device), lane_existence.to(device)
            optimizer.zero_grad()

            with autocast(is_mixed_precision):
                # To support intermediate losses for SAD
                if method == 'lstr':
                    loss, log_dict = criterion(inputs, labels, net)
                else:
                    loss, log_dict = criterion(inputs, labels, lane_existence, net, input_sizes[0])

            if is_mixed_precision:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            lr_scheduler.step()

            log_dict = reduce_dict(log_dict)
            if running_loss is None:  # Because different methods may have different values to log
                running_loss = {k: 0.0 for k in log_dict.keys()}
            for k in log_dict.keys():
                running_loss[k] += log_dict[k]
            current_step_num = int(epoch * len(loader) + i + 1)

            # Record losses
            if current_step_num % loss_num_steps == (loss_num_steps - 1):
                for k in running_loss.keys():
                    print('[%d, %d] %s: %.4f' % (epoch + 1, i + 1, k, running_loss[k] / loss_num_steps))
                    # Logging only once
                    if is_main_process():
                        writer.add_scalar(k, running_loss[k] / loss_num_steps, current_step_num)
                    running_loss[k] = 0.0

            # Record checkpoints
            if validation_loader is not None:
                if current_step_num % val_num_steps == (val_num_steps - 1) or \
                        current_step_num == num_epochs * len(loader):
                    # save_checkpoint(net=net, optimizer=optimizer, lr_scheduler=lr_scheduler,
                    #                 filename=exp_name + '_' + str(current_step_num) + '.pt')

                    test_pixel_accuracy, test_mIoU = fast_evaluate(loader=validation_loader, device=device, net=net,
                                                                   num_classes=num_classes,
                                                                   output_size=input_sizes[0],
                                                                   is_mixed_precision=is_mixed_precision)
                    if is_main_process():
                        writer.add_scalar('test pixel accuracy',
                                          test_pixel_accuracy,
                                          current_step_num)
                        writer.add_scalar('test mIoU',
                                          test_mIoU,
                                          current_step_num)
                    net.train()

                    # Record best model (straight to disk)
                    if test_mIoU > best_validation:
                        best_validation = test_mIoU
                        save_checkpoint(net=net.module if ddp else net, optimizer=optimizer,
                                        lr_scheduler=lr_scheduler, filename=exp_name + '.pt')

        epoch += 1
        print('Epoch time: %.2fs' % (time.time() - time_now))

    # For no-evaluation mode
    if validation_loader is None:
        save_checkpoint(net=net.module if ddp else net, optimizer=optimizer, lr_scheduler=lr_scheduler,
                        filename=exp_name + '.pt')


def fast_evaluate(net, device, loader, is_mixed_precision, output_size, num_classes):
    # Fast evaluation (e.g. on the validation set) by pixel-wise mean IoU
    net.eval()
    conf_mat = ConfusionMatrix(num_classes)
    with torch.no_grad():
        for image, target in tqdm(loader):
            image, target = image.to(device), target.to(device)
            with autocast(is_mixed_precision):
                output = net(image)['out']
                output = torch.nn.functional.interpolate(output, size=output_size, mode='bilinear', align_corners=True)
                conf_mat.update(target.flatten(), output.argmax(1).flatten())
        conf_mat.reduce_from_all_processes()

    acc_global, acc, iu = conf_mat.compute()
    print((
        'global correct: {:.2f}\n'
        'average row correct: {}\n'
        'IoU: {}\n'
        'mean IoU: {:.2f}'
    ).format(
        acc_global.item() * 100,
        ['{:.2f}'.format(i) for i in (acc * 100).tolist()],
        ['{:.2f}'.format(i) for i in (iu * 100).tolist()],
        iu.mean().item() * 100))

    return acc_global.item() * 100, iu.mean().item() * 100


# A unified inference function, for segmentation-based lane detection methods
@torch.no_grad()
def lane_as_segmentation_inference(net, inputs, input_sizes, gap, ppl, thresh, dataset, max_lane=0, forward=True):
    # Assume net and images are on the same device
    # images: B x C x H x W
    # Return: a list of lane predictions on each image
    outputs = net(inputs) if forward else inputs  # Support no forwarding inside this function
    prob_map = torch.nn.functional.interpolate(outputs['out'], size=input_sizes[0], mode='bilinear',
                                               align_corners=True).softmax(dim=1)
    existence_conf = outputs['lane'].sigmoid()
    existence = existence_conf > 0.5
    if max_lane != 0:  # Lane max number prior for testing
        existence, existence_conf = lane_pruning(existence, existence_conf, max_lane=max_lane)

    prob_map = prob_map.cpu().numpy()
    existence = existence.cpu().numpy()

    # Get coordinates for lanes
    lane_coordinates = []
    for j in range(existence.shape[0]):
        lane_coordinates.append(prob_to_lines(prob_map[j], existence[j], resize_shape=input_sizes[1],
                                              gap=gap, ppl=ppl, thresh=thresh, dataset=dataset))

    return lane_coordinates


# Adapted from harryhan618/SCNN_Pytorch
@torch.no_grad()
def test_one_set(net, device, loader, is_mixed_precision, input_sizes, gap, ppl, thresh, dataset,
                 method='baseline', max_lane=0, exp_name=None):
    # Predict on 1 data_loader and save predictions for the official script
    # sizes: [input size, test original size, ...]
    # max_lane = 0 -> unlimited number of lanes

    all_lanes = []
    net.eval()
    for images, filenames in tqdm(loader):
        images = images.to(device)
        with autocast(is_mixed_precision):
            if method in ['baseline', 'scnn', 'resa']:
                batch_coordinates = lane_as_segmentation_inference(net, images, input_sizes, gap, ppl, thresh, dataset,
                                                                   max_lane)
            else:
                batch_coordinates = net.inference(images, input_sizes, gap, ppl, dataset, max_lane)

        # Parse coordinates
        for j in range(len(batch_coordinates)):
            lane_coordinates = batch_coordinates[j]
            if dataset == 'culane':
                # Save each lane to disk
                dir_name = filenames[j][:filenames[j].rfind('/')]
                if not os.path.exists(dir_name):
                    os.makedirs(dir_name)
                with open(filenames[j], "w") as f:
                    for lane in lane_coordinates:
                        if lane:  # No printing for []
                            for (x, y) in lane:
                                print("{} {}".format(x, y), end=" ", file=f)
                            print(file=f)
            elif dataset == 'tusimple':
                # Save lanes to a single file
                formatted = {
                    "h_samples": [160 + y * 10 for y in range(ppl)],
                    "lanes": [[c[0] for c in lane] for lane in lane_coordinates],
                    "run_time": 0,
                    "raw_file": filenames[j]
                }
                all_lanes.append(json.dumps(formatted))
            elif dataset == 'llamas':
                # save each lane in images in xxx.lines.txt
                dir_name = filenames[j][:filenames[j].rfind('/')]
                file_path = filenames[j].replace("_color_rect", "")
                if not os.path.exists(dir_name):
                    os.makedirs(dir_name)
                with open(file_path, "w") as f:
                    for lane in lane_coordinates:
                        if lane:  # No printing for []
                            for (x, y) in lane:
                                print("{} {}".format(x, y), end=" ", file=f)
                            print(file=f)
            else:
                raise ValueError

    if dataset == 'tusimple':
        with open('./output/' + exp_name + '.json', 'w') as f:
            for lane in all_lanes:
                print(lane, end="\n", file=f)


# Adapted from harryhan618/SCNN_Pytorch
# Note that in tensors we have indices start from 0 and in annotations coordinates start at 1
def get_lane(prob_map, gap, ppl, thresh, resize_shape=None, dataset='culane'):
    """
    Arguments:
    ----------
    prob_map: prob map for single lane, np array size (h, w)
    resize_shape:  reshape size target, (H, W)
    Return:
    ----------
    coords: x coords bottom up every gap px, 0 for non-exist, in resized shape
    """

    if resize_shape is None:
        resize_shape = prob_map.shape
    h, w = prob_map.shape
    H, W = resize_shape
    coords = np.zeros(ppl)
    for i in range(ppl):
        if dataset == 'tusimple':  # Annotation start at 10 pixel away from bottom
            y = int(h - (ppl - i) * gap / H * h)
        elif dataset in ['culane', 'llamas']:  # Annotation start at bottom
            y = int(h - i * gap / H * h - 1)  # Same as original SCNN code
        # elif dataset == 'llamas':  # Annotation start at bottom
        #     y = int(h - i * gap / H * h - 1)  # Same as culane format
        else:
            raise ValueError
        if y < 0:
            break
        line = prob_map[y, :]
        id = np.argmax(line)
        if line[id] > thresh:
            coords[i] = int(id / w * W)
    if (coords > 0).sum() < 2:
        coords = np.zeros(ppl)
    return coords


# Adapted from harryhan618/SCNN_Pytorch
def prob_to_lines(seg_pred, exist, resize_shape=None, smooth=True, gap=20, ppl=None, thresh=0.3, dataset='culane'):
    """
    Arguments:
    ----------
    seg_pred: np.array size (num_classes, h, w)
    resize_shape:  reshape size target, (H, W)
    exist:   list of existence, e.g. [0, 1, 1, 0]
    smooth:  whether to smooth the probability or not
    gap: y pixel gap for sampling
    ppl:     how many points for one lane
    thresh:  probability threshold
    all_points: Whether to save all sample points or just points predicted as lane
    Return:
    ----------
    coordinates: [x, y] list of lanes, e.g.: [ [[9, 569], [50, 549]] ,[[630, 569], [647, 549]] ]
    """
    if resize_shape is None:
        resize_shape = seg_pred.shape[1:]  # seg_pred (num_classes, h, w)
    _, h, w = seg_pred.shape
    H, W = resize_shape
    coordinates = []

    if ppl is None:
        ppl = round(H / 2 / gap)

    for i in range(1, seg_pred.shape[0]):
        prob_map = seg_pred[i, :, :]
        if exist[i - 1]:
            if smooth:
                prob_map = cv2.blur(prob_map, (9, 9), borderType=cv2.BORDER_REPLICATE)
            coords = get_lane(prob_map, gap, ppl, thresh, resize_shape, dataset=dataset)
            if coords.sum() == 0:
                continue
            if dataset == 'tusimple':  # Invalid sample points need to be included as negative value, e.g. -2
                coordinates.append([[coords[j], H - (ppl - j) * gap] if coords[j] > 0 else [-2,  H - (ppl - j) * gap]
                                    for j in range(ppl)])
            elif dataset in ['culane', 'llamas']:
                coordinates.append([[coords[j], H - j * gap - 1] for j in range(ppl) if coords[j] > 0])
            # elif dataset == 'llamas':
            #     coordinates.append([[coords[j], H - j * gap - 1] for j in range(ppl) if coords[j] > 0])
            else:
                raise ValueError

    return coordinates


def build_lane_detection_model(args, num_classes):
    scnn = True if args.method == 'scnn' else False
    spatial_conv = args.method if args.method in ['scnn', 'resa'] else None
    if args.dataset == 'tusimple' and args.backbone == 'erfnet':
        net = erfnet_tusimple(num_classes=num_classes, scnn=scnn)
    elif args.method == 'lstr':
        if (hasattr(args, 'state') and args.state == 1) or (hasattr(args, 'val_num_steps') and args.val_num_steps != 0):
            print('Fast validation not supported for this method!')
            raise ValueError
        num_classes_max = 7 if args.dataset in ['culane', 'llamas', 'tusimple'] else num_classes
        net = lstr_resnet(num_classes_max=num_classes_max, backbone_name=args.backbone,
                          expansion=1 if args.dataset == 'tusimple' else 2,
                          aux_loss=True if hasattr(args, 'state') and args.state == 0 else False)
    elif args.dataset == 'culane' and args.backbone == 'erfnet':
        net = erfnet_culane(num_classes=num_classes, scnn=scnn)
    elif args.dataset == 'culane' and args.backbone == 'vgg16':
        net = vgg16_culane(num_classes=num_classes, scnn=scnn)
    elif args.dataset == 'tusimple' and args.backbone == 'vgg16':
        net = vgg16_tusimple(num_classes=num_classes, scnn=scnn)
    elif args.dataset == 'tusimple' and 'resnet' in args.backbone:
        net = resnet_tusimple(num_classes=num_classes, spatial_conv=spatial_conv, backbone_name=args.backbone)
    elif args.dataset == 'culane' and 'resnet' in args.backbone:
        net = resnet_culane(num_classes=num_classes, spatial_conv=spatial_conv, backbone_name=args.backbone)
    elif args.dataset == 'tusimple' and args.backbone == 'enet':
        net = enet_tusimple(num_classes=num_classes, encoder_only=args.encoder_only,
                            continue_from=args.continue_from)
    elif args.dataset == 'culane' and args.backbone == 'enet':
        net = enet_culane(num_classes=num_classes, encoder_only=args.encoder_only,
                          continue_from=args.continue_from)
    elif args.dataset == 'llamas' and args.backbone == 'erfnet':
        net = erfnet_llamas(num_classes=num_classes, scnn=scnn)
    elif args.dataset == 'llamas' and args.backbone == 'vgg16':
        net = vgg16_llamas(num_classes=num_classes, scnn=scnn)
    elif args.dataset == 'llamas' and 'resnet' in args.backbone:
        net = resnet_llamas(num_classes=num_classes, spatial_conv=spatial_conv, backbone_name=args.backbone)
    else:
        raise ValueError

    return net
