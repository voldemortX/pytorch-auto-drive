import yaml
import argparse
from utils.all_utils_landec import build_lane_detection_model as build_lane_model
from utils.all_utils_semseg import build_segmentation_model, load_checkpoint
from tools.profiling_utils import init_lane, init_seg, speed_evaluate_real, speed_evaluate_simple, model_profile
import torch

if __name__ == '__main__':
    # Settings
    parser = argparse.ArgumentParser(description='PyTorch Auto-drive')
    parser.add_argument('--height', type=int, default=288,
                        help='Image input height (default: 288)')
    parser.add_argument('--width', type=int, default=800,
                        help='Image input width (default: 800)')
    parser.add_argument('--dataset', type=str, default='tusimple',
                        help='Profile on TuSimple (tusimple) / CULane (culane) (default: tusimple)')
    parser.add_argument('--method', type=str, default='baseline',
                        help='method selection (lstr/scnn/resa/sad/baseline) (default: baseline)')
    parser.add_argument('--backbone', type=str, default='erfnet',
                        help='backbone selection (erfnet/enet/vgg16/resnet18s/resnet18/resnet34/resnet50/resnet101)'
                             '(default: erfnet)')
    parser.add_argument('--task', type=str, default='lane',
                        help='task selection (lane/seg)')
    parser.add_argument('--mode', type=str, default='simple',
                        help='Profiling mode (simple/real)')
    parser.add_argument('--model', type=str, default='deeplabv3',
                        help='Model selection (fcn/erfnet/deeplabv2/deeplabv3/enet) (default: deeplabv3)')
    parser.add_argument('--times', type=int, default=1,
                        help='Select test times')
    parser.add_argument('--encoder-only', action='store_true', default=False,
                        help='Only train the encoder. ENet trains encoder and decoder separately (default: False)')
    parser.add_argument('--continue-from', type=str, default=None,
                        help='Continue training from a previous checkpoint')
    args = parser.parse_args()
    lane_need_interpolate = ['baseline', 'scnn', 'sad', 'resa']
    seg_need_interpolate = ['fcn', 'deeplabv2', 'deeplabv3']
    with open('configs.yaml', 'r') as f:  # Safer and cleaner than box/EasyDict
        configs = yaml.load(f, Loader=yaml.Loader)
    mean = configs['GENERAL']['MEAN']
    std = configs['GENERAL']['STD']
    if args.task == 'lane':
        num_classes = configs[configs['LANE_DATASETS'][args.dataset]]['NUM_CLASSES']
        count_interpolate = False
        if args.method in lane_need_interpolate:
            count_interpolate = True
        if torch.cuda.is_available():
            device = torch.device('cuda:0')
        net = build_lane_model(args, num_classes)
        net.to(device)
        print(device)
        macs, params = model_profile(net, args.height, args.width, device)
        print('FLOPs(G): {: .2f}'.format(2 * macs / 1e9))
        print('Number of parameters: {: .2f}'.format(params / 1e6))
        print('Profiling, please clear your GPU memory before doing this.')
        if args.mode == 'simple':
            dummy = torch.ones((1, 3, args.height, args.width))
            print(dummy.dtype)
            fps = []
            for i in range(0, args.times):
                fps.append(speed_evaluate_simple(net=net, device=device, dummy=dummy, num=300,
                                                 count_interpolate=count_interpolate))
            print('GPU FPS: {: .2f}'.format(max(fps)))
        elif args.mode == 'real' and args.dataset in configs['LANE_DATASETS'].keys():
            load_checkpoint(net=net, optimizer=None, lr_scheduler=None, filename=args.continue_from)
            base = configs[configs['LANE_DATASETS'][args.dataset]]['BASE_DIR']
            val_loader = init_lane(dataset=args.dataset, input_sizes=(args.height, args.width), mean=mean, std=std,
                                   base=base)
            fps = []
            gpu_fps = []
            for i in range(0, args.times):
                fps_item, gpu_fps_item = speed_evaluate_real(net=net, device=device, loader=val_loader, num=300,
                                                             count_interpolate=count_interpolate)
                fps.append(fps_item)
                gpu_fps.append(gpu_fps_item)
            print('Real FPS: {: .2f}'.format(max(fps)))
            print('GPU FPS: {: .2f}'.format(max(gpu_fps)))
        else:
            raise ValueError
    elif args.task == 'seg':
        num_classes = configs[configs['SEGMENTATION_DATASETS'][args.dataset]]['NUM_CLASSES']
        input_sizes = (args.height, args.width)
        city_aug = 0
        count_interpolate = False
        if args.backbone in seg_need_interpolate:
            count_interpolate = True
        if torch.cuda.is_available():
            device = torch.device('cuda:0')
        print(device)
        net, city_aug, _, _ = build_segmentation_model(configs, args, num_classes, city_aug, input_sizes)
        net.to(device)
        macs, params = model_profile(net, args.height, args.width, device)
        print('FLOPs(G): {: .2f}'.format(2 * macs / 1e9))
        print('Number of parameters: {: .2f}'.format(params / 1e6))
        print('Profiling, please clear your GPU memory before doing this.')
        if args.mode == 'simple':
            dummy = torch.ones((1, 3, args.height, args.width))
            fps = []
            for i in range(0, args.times):
                fps.append(speed_evaluate_simple(net=net, device=device, dummy=dummy, num=300,
                                                 count_interpolate=count_interpolate))
            print('GPU FPS: {: .2f}'.format(max(fps)))
        elif args.mode == 'real' and args.dataset in configs['SEGMENTATION_DATASETS'].keys():
            load_checkpoint(net=net, optimizer=None, lr_scheduler=None, filename=args.continue_from)
            base = configs[configs['SEGMENTATION_DATASETS'][args.dataset]]['BASE_DIR']
            train_label_id_map = configs[configs['SEGMENTATION_DATASETS'][args.dataset]]['LABEL_ID_MAP'] if \
                'LABEL_ID_MAP' in configs[configs['SEGMENTATION_DATASETS'][args.dataset]].keys() else \
                configs['CITYSCAPES']['LABEL_ID_MAP']
            val_loader = init_seg(dataset=args.dataset, input_sizes=(args.height, args.width), mean=mean,
                                  std=std, test_base=base, city_aug=city_aug, test_label_id_map=train_label_id_map)
            fps = []
            gpu_fps = []
            for i in range(0, args.times):
                fps_item, gpu_fps_item = speed_evaluate_real(net=net, device=device, loader=val_loader, num=300,
                                                             count_interpolate=count_interpolate)
                fps.append(fps_item)
                gpu_fps.append(gpu_fps_item)
            print('Real FPS: {: .2f}'.format(max(fps)))
            print('GPU FPS: {: .2f}'.format(max(gpu_fps)))
        else:
            raise ValueError
