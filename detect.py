# -*- coding: utf-8 -*-

"""
@Author: BQH
@File: detect.py
@Description:
@Date: 2021/5/9
"""

import os
import tqdm
import numpy as np
import math
import cv2
import time
import torch
from PIL import Image
from PIL import ImageOps
import matplotlib.pyplot as plt

from config import cfg
from nets.hourglass import get_hourglass

from utils.utils import load_model
from utils.post_process import ctdet_decode
from tools.data_augment import _img_nomal

class_names = ['BG', 'Hip', 'Knee', 'Ankle']


def run_time(func):
    def wrapper(self, *args, **kw):
        local_time = time.time()
        bbox = func(self, *args, **kw)
        print('current Function [%s] run time is %.2f' % (func.__name__ , time.time() - local_time))
        return bbox
    return wrapper


def py_cpu_nms(dets, thresh):
    """Pure Python NMS baseline."""
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]

    return keep


def plot_boxes_cv2(img, boxes, savename=None, class_names=None, color=None):
    img = np.copy(img)
    if len(img.shape) < 3:
        img = np.stack((img,)*3, axis=-1)
    colors = np.array([[1, 0, 1], [0, 0, 1], [0, 1, 1], [0, 1, 0], [1, 1, 0], [1, 0, 0]], dtype=np.float32)

    def get_color(c, x, max_val):
        ratio = float(x) / max_val * 5
        i = int(math.floor(ratio))
        j = int(math.ceil(ratio))
        ratio = ratio - i
        r = (1 - ratio) * colors[i][c] + ratio * colors[j][c]
        return int(r * 255)

    for i in range(len(boxes)):
        box = boxes[i]
        x1 = int(box[0])
        y1 = int(box[1])
        x2 = int(box[2])
        y2 = int(box[3])

        if color:
            rgb = color
        else:
            rgb = (255, 0, 0)
        if len(box) >= 6 and class_names:
            cls_conf = box[4]
            cls_id = int(box[5])
            test = ('%s: %.2f' % (class_names[cls_id], cls_conf))
            classes = len(class_names)
            offset = cls_id * 123457 % classes
            red = get_color(2, offset, classes)
            green = get_color(1, offset, classes)
            blue = get_color(0, offset, classes)
            if color is None:
                rgb = (red, green, blue)
            cx = x1 + 5
            cy = y1 + 12
            img = cv2.putText(img, test, (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.8, rgb, 1)
        img = cv2.rectangle(img, (x1, y1), (x2, y2), rgb, 2)
    if savename:
        print("save plot results to %s" % savename)
        cv2.imwrite(savename, img)
    return img


class Detector(object):
    def __init__(self, model_path, confidence_threshold=0.7):
        self.net = get_hourglass(cfg.arch, num_classes=cfg.num_classes, is_training=False)
        load_model(self.net, model_path)
        self.net = self.net.to(cfg.device)
        self.net.eval()

        self.nms_threshold = 0.4
        self.confidence_threshold = confidence_threshold

    @run_time
    def predict(self, images: list):
        """
        要求输入的一组图片大小一致
        """
        input_x, padding_info, resize = self.preprocess(images)
        output = self.net(input_x)[-1]
        dets = self.post_processing(output, padding_info, resize)
        return dets

    def get_img_ratio(self, img_size, target_size):
        img_rate = np.max(img_size) / np.min(img_size)
        target_rate = np.max(target_size) / np.min(target_size)
        if img_rate > target_rate:
            # 按长边缩放
            ratio = max(target_size) / max(img_size)
        else:
            ratio = min(target_size) / min(img_size)
        return ratio

    def PIL_img_scale_pad_only_image(self, img, outsize=None):
        w, h = img.size
        target_w, target_h = outsize[0], outsize[1]
        ratio = self.get_img_ratio([w, h], outsize)  # outsize / max(w, h)
        ow, oh = round(w * ratio), round(h * ratio)
        img = img.resize((ow, oh), Image.BILINEAR)
        dh, dw = target_h - oh, target_w - ow
        top, bottom = dh // 2, dh - (dh // 2)
        left, right = dw // 2, dw - (dw // 2)
        img = ImageOps.expand(img, border=(left, top, right, bottom), fill=0)  # 左 顶 右 底 顺时针
        return img, ratio, [left, top, left, top]

    def preprocess(self, images):
        net_input = []
        for image in images:
            img, resize, padding_info = self.PIL_img_scale_pad_only_image(Image.fromarray(image),
                                                                          outsize=cfg.net_input_size)
            img = _img_nomal(np.array(img, dtype=np.uint8))
            net_input.append([img])

        # padding_info = torch.Tensor(padding_info)
        net_input = np.concatenate(net_input, axis=0)
        net_input = torch.from_numpy(net_input)
        net_input = net_input.to(cfg.device)
        # padding_info = padding_info.to(cfg.device)
        return net_input, padding_info, resize

    def post_processing(self, output, padding_info, resize):
        """
        返回值：[batch, n, 6]
        格式：x1,y1,x2,y2,confidence,class_id
        """
        dets = ctdet_decode(*output)
        res = []

        for det in dets:
            labels = det[:, -1]
            conf = torch.sigmoid(det[:, 4])
            box_array = det[:, :4] * cfg.down_ratio  # 还原到网络的输入图像尺寸

            labels = labels.detach().cpu().numpy()
            conf = conf.detach().cpu().numpy()
            box_array = box_array.detach().cpu().numpy()

            box_array = (box_array - padding_info) / resize  # 还原到原始尺寸
            bboxes = np.zeros((0, 6))
            for class_index in range(1, len(class_names)):
                cls_argwhere = labels == class_index
                ll_max_id = labels[cls_argwhere].reshape(-1, 1)
                scores = conf[cls_argwhere].reshape(-1, 1)
                box_array_new = box_array[cls_argwhere, :]
                bboxes = np.vstack((bboxes, np.hstack((box_array_new, scores, ll_max_id))))
            res.append(bboxes)
        return res


if __name__ == '__main__':
    weights_file = r'ckpt/centernet_Epoch76_map0.9232395168098128.pth'
    img_path = r'F:\code\Data\DectDataset\case_001\coronal\221.jpg'
    img = cv2.imread(img_path)
    detector = Detector(weights_file, confidence_threshold=0.6)
    dets = detector.predict([img])
    det_img = plot_boxes_cv2(img, dets[0], class_names=class_names)
    plt.imshow(det_img)
    plt.show()