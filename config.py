# -*- coding: utf-8 -*-

"""
@Author: BQH
@File: config.py
@Description:
@Date: 2021/5/9
"""
import os
import time
from easydict import EasyDict

cfg = EasyDict()

_BASE_DIR = os.path.dirname(os.path.abspath(__file__))

cfg.root_dir = _BASE_DIR
cfg.device = 'cuda'
cfg.num_gpu = 2
cfg.net_input_size = [512, 256]  # w,h格式
cfg.down_ratio = 4
cfg.arch ='large_hourglass'
cfg.pretrained_weights = None
cfg.batch_size = 2
cfg.num_classes = 3 + 1

save_dir_name = time.strftime('%Y%m%d%M%I%S')
cfg.log_dir = os.path.join(cfg.root_dir, 'logs', save_dir_name)
cfg.ckpt_dir = os.path.join(cfg.root_dir, 'ckpt', save_dir_name)

cfg.train_label = os.path.join(cfg.root_dir, 'datasets/train.md')
cfg.val_label = os.path.join(cfg.root_dir, 'datasets/val.md')