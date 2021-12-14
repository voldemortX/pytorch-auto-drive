import os
import torch
try:
    import ujson as json
except ImportError:
    import json
from tqdm import tqdm
if torch.__version__ >= '1.6.0':
    from torch.cuda.amp import autocast
else:
    from ..torch_amp_dummy import autocast

from .base import BaseTester
from ..seg_utils import ConfusionMatrix
from ..lane_det_utils import lane_as_segmentation_inference


class LaneDetTester(BaseTester):
    image_sets = ['valfast', 'test', 'val']

    def __init__(self, cfg, args):
        super().__init__(cfg, args)
        self.fast_eval = True if args.state == 1 else False

    def run(self):
        if self.fast_eval:
            _, x = self.fast_evaluate(self.model, self.device, self.dataloader,
                                      self._cfg['mixed_precision'], self._cfg['input_size'], self._cfg['num_classes'])
            self.write_mp_log('log.txt', self._cfg['exp_name'] + ' validation: ' + str(x) + '\n')
        else:
            self.test_one_set(self.model, self.device, self.dataloader, self._cfg['mixed_precision'],
                              [self._cfg['input_size'], self._cfg['original_size']],
                              self._cfg['gap'], self._cfg['ppl'], self._cfg['thresh'],
                              self._cfg['dataset_name'], self._cfg['seg'], self._cfg['max_lane'], self._cfg['exp_name'])

    @staticmethod
    @torch.no_grad()
    def test_one_set(net, device, loader, mixed_precision, input_sizes, gap, ppl, thresh, dataset,
                     seg, max_lane=0, exp_name=None):
        # Adapted from harryhan618/SCNN_Pytorch
        # Predict on 1 data_loader and save predictions for the official script
        # sizes: [input size, test original size, ...]
        # max_lane = 0 -> unlimited number of lanes

        all_lanes = []
        net.eval()
        for images, filenames in tqdm(loader):
            images = images.to(device)
            with autocast(mixed_precision):
                if seg:
                    batch_coordinates = lane_as_segmentation_inference(net, images,
                                                                       input_sizes, gap, ppl, thresh, dataset, max_lane)
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

    @staticmethod
    @torch.no_grad()
    def fast_evaluate(net, device, loader, mixed_precision, output_size, num_classes):
        # Fast evaluation (e.g. on the validation set) by pixel-wise mean IoU
        net.eval()
        conf_mat = ConfusionMatrix(num_classes)
        for image, target in tqdm(loader):
            image, target = image.to(device), target.to(device)
            with autocast(mixed_precision):
                output = net(image)['out']
                output = torch.nn.functional.interpolate(output, size=output_size,
                                                         mode='bilinear', align_corners=True)
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
