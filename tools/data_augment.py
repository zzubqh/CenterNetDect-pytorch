import random
import numpy as np
from PIL import Image
from PIL import ImageOps
import matplotlib.pyplot as plt

from .aug_strategy import imgaug_boxes
from .aug_strategy import imgaug_image
from .aug_strategy import pipe_sequential_rotate
from .aug_strategy import pipe_sequential_translate
from .aug_strategy import pipe_sequential_scale
from .aug_strategy import pipe_someof_flip
from .aug_strategy import pipe_sometimes_hsv
from .aug_strategy import pipe_someof_blur
from .aug_strategy import pipe_sometimes_mpshear
from .aug_strategy import pipe_someone_contrast
from .aug_strategy import plt_box_on_img


def _img_nomal(image):
    img = image.astype(np.float32)
    # BGR图像
    img -= np.array([3.035530335600759, 3.022813865832708, 3.0538134711893106])
    img /= np.array([21.871503077765304, 22.0234010758844, 21.765863316862184])
    return img.transpose(2, 0, 1)


def get_data_aug_pipe():
    pipe_aug = []
    if random.random() > 0.5:

        aug_list = [pipe_sequential_rotate, pipe_sequential_scale, pipe_sequential_translate, pipe_someof_blur,
                    pipe_someof_flip, pipe_sometimes_hsv, pipe_sometimes_mpshear, pipe_someone_contrast]
        index = np.random.choice(a=[0, 1, 2, 3, 4, 5, 6, 7],
                                 p=[0.05, 0.25, 0.20, 0.25, 0.10, 0.05, 0.05, 0.05])
        if (index == 0 or index == 4 or index == 5) and random.random() < 0.5:  # 会稍微削弱旋转 但是会极大增强其他泛化能力
            index2 = np.random.choice(a=[1, 2, 3], p=[0.4, 0.3, 0.3])
            pipe_aug = [aug_list[index], aug_list[index2]]
        else:
            pipe_aug = [aug_list[index]]
    return pipe_aug


def get_img_ratio(img_size, target_size):
    img_rate = np.max(img_size) / np.min(img_size)
    target_rate = np.max(target_size) / np.min(target_size)
    if img_rate > target_rate:
        # 按长边缩放
        ratio = max(target_size) / max(img_size)
    else:
        ratio = min(target_size) / min(img_size)
    return ratio


def PIL_img_scale_pad(img, boxes, Interpolation=Image.BILINEAR, outsize=None):
    if outsize is None:
        outsize = [300, 300]

    def update_bbox(bbox):
        bbox = bbox.copy()
        bbox *= ratio
        bbox[0], bbox[1], bbox[2], bbox[3] = left + bbox[0], top + bbox[1], left + bbox[2], top + bbox[3]
        return bbox

    w, h = img.size
    target_w, target_h = outsize[0], outsize[1]
    ratio = get_img_ratio([w, h], outsize)  # outsize / max(w, h)
    ow, oh = round(w * ratio), round(h * ratio)
    img = img.resize((ow, oh), Interpolation)
    dh, dw = target_h - oh, target_w - ow
    top, bottom = dh // 2, dh - (dh // 2)
    left, right = dw // 2, dw - (dw // 2)
    img = ImageOps.expand(img, border=(left, top, right, bottom), fill=0)  # 左 顶 右 底 顺时针
    return img, np.array([update_bbox(i) for i in boxes])


def _sync_transform(img, boxes, outsize=None):
    img, boxes = PIL_img_scale_pad(Image.fromarray(img), boxes, outsize=outsize)
    img = np.asarray(img)
    pipe_aug = get_data_aug_pipe()
    if len(pipe_aug):
        for i in pipe_aug:
            img, boxes = imgaug_boxes(img, boxes, i)
    return np.array(img, dtype=np.uint8), boxes


def PIL_img_scale_pad_only_image(img, Interpolation=Image.BILINEAR, outsize=None):
    if outsize is None:
        outsize = [300, 300]
    w, h = img.size
    target_w, target_h = outsize[0], outsize[1]
    ratio = get_img_ratio([w, h], outsize)  # outsize / max(w, h)
    ow, oh = round(w * ratio), round(h * ratio)
    img = img.resize((ow, oh), Interpolation)
    dh, dw = target_h - oh, target_w - ow
    top, bottom = dh // 2, dh - (dh // 2)
    left, right = dw // 2, dw - (dw // 2)
    img = ImageOps.expand(img, border=(left, top, right, bottom), fill=0)  # 左 顶 右 底 顺时针
    return img


def _sync_transform_only_image(img, outsize=None):
    img = PIL_img_scale_pad_only_image(Image.fromarray(img), outsize=outsize)
    img = np.asarray(img)
    pipe_aug = get_data_aug_pipe()
    if len(pipe_aug):
        for i in pipe_aug:
            img = imgaug_image(img, i)
    return np.array(img, dtype=np.uint8)


class preproc(object):
    def __init__(self, net_input_size):
        self.net_input_size = net_input_size

    def show(self, im1, boxes1, im2, boxes2):
        fig, ax = plt.subplots(1, 2)
        ax[0].imshow(im1)
        ax[1].imshow(im2)
        for (b1, b2) in zip(boxes1, boxes2):
            plt_box_on_img(ax[0], b1)
            plt_box_on_img(ax[1], b2)
        plt.show()

    def __call__(self, image, targets, is_show=False):
        assert targets.shape[0] > 0, "this image does not have gt"

        boxes = targets[:, :4].copy()
        labels = targets[:, -1].copy()
        image_t, boxes_t = _sync_transform(image, boxes.copy(), self.net_input_size)
        labels_t = labels

        height, width, _ = image_t.shape
        if is_show:
            self.show(image, boxes, image_t, boxes_t)
        image_t = _img_nomal(image_t)
        # boxes_t[:, 0::2] /= width
        # boxes_t[:, 1::2] /= height
        labels_t = np.expand_dims(labels_t, 1)
        targets_t = np.hstack((boxes_t, labels_t))

        return image_t, targets_t