from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# import os
# import logging

# from config import cfg
# from config import update_config
from config.default import _C as cfg
from config.default import update_config
import argparse

# import torch
# import torch.nn as nn

def parse_args():
    parser = argparse.ArgumentParser(description='Train keypoints network')
    # general
    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        required=False,
                        default= 'D:\CV-Project\pytorch-pose-hg-3d\experiments\mpii\hrnet\w32_256x256_adam_lr1e-3.yaml',
                        type=str)

    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)

    # philly
    parser.add_argument('--modelDir',
                        help='model directory',
                        type=str,
                        default='')
    parser.add_argument('--logDir',
                        help='log directory',
                        type=str,
                        default='')
    parser.add_argument('--dataDir',
                        help='data directory',
                        type=str,
                        default='')
    parser.add_argument('--prevModelDir',
                        help='prev Model directory',
                        type=str,
                        default='')

    args = parser.parse_args()

    return args


def main():
    args = parse_args()
    update_config(cfg, args)
    extra = cfg.MODEL.EXTRA
    print(cfg.MODEL.NAME)
    print(extra.FINAL_CONV_KERNEL)
    print(cfg.MODEL.EXTRA.STAGE2.NUM_CHANNELS)




if __name__ == '__main__':
    main()