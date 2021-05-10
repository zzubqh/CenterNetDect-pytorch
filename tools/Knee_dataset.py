# -*- coding:utf-8 -*-
# name: Knee_dataset
# author: bqh
# datetime:2021/04/19 15:26
# =========================

import os
import os.path
import torch
import cv2
import numpy as np
import torch.utils.data as data
from PIL import Image
import re
import math

from tools.data_augment import _sync_transform_only_image
from tools.data_augment import _img_nomal
from tools.data_augment import PIL_img_scale_pad
from tools.data_augment import preproc

from utils.image import get_border, get_affine_transform, affine_transform, color_aug
from utils.image import draw_umich_gaussian, gaussian_radius

class_names = ['BG', 'Hip', 'Knee', 'Ankle']
BONE_IDS = [0, 1, 2, 3]


class KneeDataset(data.Dataset):
    def __init__(self, label_path, cfg, train=True):
        super(KneeDataset, self).__init__()
        positive_labels = []
        self.config = cfg
        img_size = cfg.net_input_size  # [w, h]格式
        with open(label_path, 'r', encoding='utf-8') as rf:
            for line in rf:
                sample = {}
                items = line.strip().split(' ')
                sample['file'] = items[0]
                sample['boxes'] = []
                for item in items[1:]:
                    bbox_xy = [int(float(j)) for j in item.split(',')]
                    sample['boxes'].append(bbox_xy)
                if len(sample['boxes']) > 6:
                    print(sample['file'])
                positive_labels.append(sample)

        self.train = train
        self.samples = [i for i in np.random.permutation(positive_labels)]
        print("加载样本:{},成功加载:{}".format(len(positive_labels), len(self.samples)))
        self.preproc = preproc(cfg.net_input_size)

        self.down_ratio = 4
        self.img_size = {'h': img_size[1], 'w': img_size[0]}
        self.fmap_size = {'h': img_size[1] // self.down_ratio, 'w': img_size[0] // self.down_ratio}
        self.rand_scales = np.arange(0.6, 1.4, 0.1)
        self.gaussian_iou = 0.7
        self.num_samples = len(self.samples)
        self.max_objs = 6  # 冠状面最多6个
        self.num_classes = cfg.num_classes

    def __getitem__(self, index):
        if not self.train:
            return self._get_val_item(index)
        sample = self.samples[index]
        im_file = sample["file"]
        img = cv2.imread(im_file)
        if img is None:
            print(im_file)
        height, width, _ = img.shape
        boxes = sample["boxes"]
        annotations = np.zeros((0, 5))

        if len(boxes) == 0:
            img = _sync_transform_only_image(img, outsize=self.target_size)
            img = _img_nomal(img)
            return torch.from_numpy(img), annotations

        hmap = np.zeros((self.num_classes, self.fmap_size['h'], self.fmap_size['w']), dtype=np.float32)  # heatmap
        w_h_ = np.zeros((self.max_objs, 2), dtype=np.float32)  # width and height
        regs = np.zeros((self.max_objs, 2), dtype=np.float32)  # regression
        inds = np.zeros((self.max_objs,), dtype=np.int64)
        ind_masks = np.zeros((self.max_objs,), dtype=np.uint8)

        # 数据增强
        for k, box in enumerate(boxes):
            annotation = np.zeros((1, 5), dtype=np.int)
            # bbox
            annotation[0, 0] = box[0]  # x1
            annotation[0, 1] = box[1]  # y1
            annotation[0, 2] = box[2]  # x2
            annotation[0, 3] = box[3]  # y2
            annotation[0, 4] = box[4]  # class_id
            annotations = np.append(annotations, annotation, axis=0)
        target = np.array(annotations)
        img, target = self.preproc(img, target)

        # 生成网络所需的热力图
        bboxes = target[:, :4] / self.down_ratio
        labels = target[:, -1].astype(np.int)
        for k, (bbox, label) in enumerate(zip(bboxes, labels)):
            h, w = bbox[3] - bbox[1], bbox[2] - bbox[0]
            if h > 0 and w > 0:
                obj_c = np.array([(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2], dtype=np.float32)
                obj_c_int = obj_c.astype(np.int32)
                radius = max(0, int(gaussian_radius((math.ceil(h), math.ceil(w)), self.gaussian_iou)))
                draw_umich_gaussian(hmap[label], obj_c_int, radius)
                w_h_[k] = 1. * w, 1. * h
                regs[k] = obj_c - obj_c_int
                inds[k] = obj_c_int[1] * self.fmap_size['w'] + obj_c_int[0]
                ind_masks[k] = 1

        return {'image': img, 'hmap': hmap, 'w_h_': w_h_, 'regs': regs, 'inds': inds, 'ind_masks': ind_masks}

    def _get_val_item(self, index):
        sample = self.samples[index]
        img_path = sample["file"]
        bboxes_with_cls_id = np.array(sample['boxes'], dtype=np.float)
        image = cv2.imread(img_path)
        img, bbox = PIL_img_scale_pad(Image.fromarray(image), bboxes_with_cls_id[..., :4],
                                      outsize=self.config.net_input_size)
        img = _img_nomal(np.array(img, dtype=np.uint8))

        num_objs = len(bboxes_with_cls_id)
        target = {}
        # boxes to coco format
        boxes = bbox / self.down_ratio
        boxes[..., 2:] = boxes[..., 2:] - boxes[..., :2]  # box width, box height
        target['boxes'] = torch.as_tensor(boxes, dtype=torch.float32)
        target['labels'] = torch.as_tensor(bboxes_with_cls_id[..., -1].flatten(), dtype=torch.int64)
        target['image_id'] = torch.tensor([get_image_id(img_path)])
        target['area'] = (target['boxes'][:, 3]) * (target['boxes'][:, 2])
        target['iscrowd'] = torch.zeros((num_objs,), dtype=torch.int64)
        return torch.from_numpy(img), target

    def __len__(self):
        return self.num_samples


def get_image_id(filename: str):
    view_dict = {"coronal": '1', "sagittal": '2'}
    case_pattern = re.compile(r'case_\d{3}')
    view_pattern = re.compile(r'(sagittal/\d*.jpg)|(coronal/\d*.jpg)')
    number_pattern = re.compile(r'\d')

    case_name = ''.join(re.findall(case_pattern, filename))
    view_img_id = ''.join(re.findall(view_pattern, filename)[0])

    case_id = ''.join(re.findall(number_pattern, case_name))
    img_id = ''.join(re.findall(number_pattern, view_img_id))

    view_name = os.path.split(view_img_id)[0]
    str_id = case_id + view_dict[view_name] + img_id
    img_id = int(f"{int(str_id):09d}")
    return img_id


def detection_collate(batch):
    targets = []
    imgs = []
    for _, sample in enumerate(batch):
        if not sample is None:
            for _, tup in enumerate(sample):
                if torch.is_tensor(tup):
                    imgs.append(tup)
                elif isinstance(tup, type(np.empty(0))):
                    annos = torch.from_numpy(tup).float()
                    targets.append(annos)
        else:
            print("none data")

    return (torch.stack(imgs, 0), targets)


def collate_fn(batch):
    return tuple(zip(*batch))