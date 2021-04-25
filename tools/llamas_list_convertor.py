import argparse
from llamas_utils import generate_spline_annotation

if __name__ == '__main__':
    # parser = argparse.ArgumentParser(description='PyTorch Auto-drive')
    # parser.add_argument('--annotation-type', type=str, default='tusimple',
    #                     help='')
    generate_spline_annotation()
















"""
验证图像和label的匹配：
            img_num = images_list[idx].split('/')[-1].split('_')[0:2]
            json_num = json_list[idx].split('/')[-1].split('_')
            json_num[1] = json_num[1].split('.')[0]

            if img_num[0] != json_num[0] or img_num[1] != json_num[1]:
                count += 1
"""
