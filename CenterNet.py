# -*- coding: utf-8 -*-

"""
@Author: BQH
@File: CenterNet.py
@Description:
@Date: 2021/5/8
"""
import os
import tqdm
import numpy as np
import time
import torch
import torch.optim as optim

from nets.hourglass import get_hourglass

from utils.utils import _tranpose_and_gather_feature, load_model
from tools.tv_reference.coco_utils import convert_to_coco_api
from tools.tv_reference.coco_eval import CocoEvaluator

from utils.losses import _neg_loss, _reg_loss
from utils.post_process import ctdet_decode


class CenterNet(object):
    def __init__(self, cfg):
        self.cfg = cfg
        self.model = get_hourglass(cfg.arch, num_classes=cfg.num_classes)

        if cfg.pretrained_weights is not None:
            weight_file = os.path.join(cfg.save_folder, cfg.pretrained_weights)
            load_model(self.model, weight_file)
            print("load pretrain mode:{}".format(weight_file))

        if cfg.num_gpu > 1 and torch.cuda.is_available():
            self.model = torch.nn.DataParallel(self.model).cuda()
        else:
            self.model = self.model.to(cfg.device)
        self.save_folder = cfg.ckpt_dir
        self.optim = optim.Adam(self.model.parameters(), lr=cfg.lr)
        self.scheduler = optim.lr_scheduler.ExponentialLR(self.optim, gamma=0.99)

    def train(self, data_counts, data_loader, eval_loder, n_epochs):
        max_map = 0.28
        for epoch in range(n_epochs):
            evaluator = self.train_epoch(data_counts, data_loader, eval_loder, epoch, n_epochs)
            stats = evaluator.coco_eval['bbox'].stats
            eval_map = stats[0]
            if eval_map > max_map:
                max_map = eval_map
                ckpt_path = os.path.join(self.save_folder, 'centernet_Epoch{0}_map{1}.pth'.format(epoch, max_map))
                torch.save(self.model.state_dict(), ckpt_path)
                print('weights {0} saved success!'.format(ckpt_path))
            self.scheduler.step()

    def train_epoch(self, data_counts, data_loader, eval_loder, epoch, n_epochs):
        with tqdm.tqdm(total=data_counts, desc=f'Epoch {epoch}/{n_epochs}', unit='img', ncols=150) as pbar:
            step = 0
            for batch in data_loader:
                step += 1
                load_t0 = time.time()
                for k in batch:
                    batch[k] = batch[k].to(device=self.cfg.device, non_blocking=True)

                outputs = self.model(batch['image'])
                hmap, regs, w_h_ = zip(*outputs)
                regs = [_tranpose_and_gather_feature(r, batch['inds']) for r in regs]
                w_h_ = [_tranpose_and_gather_feature(r, batch['inds']) for r in w_h_]

                hmap_loss = _neg_loss(hmap, batch['hmap'])
                reg_loss = _reg_loss(regs, batch['regs'], batch['ind_masks'])
                w_h_loss = _reg_loss(w_h_, batch['w_h_'], batch['ind_masks'])
                loss = hmap_loss + 1 * reg_loss + 0.1 * w_h_loss

                self.model.zero_grad()
                loss.backward()
                self.optim.step()

                load_t1 = time.time()
                batch_time = load_t1 - load_t0
                pbar.set_postfix(**{'hmap_loss': hmap_loss.item(),
                                    'reg_loss': reg_loss.item(),
                                    'w_h_loss': w_h_loss.item(),
                                    'LR': self.optim.param_groups[0]['lr'],
                                    'Batchtime': batch_time})
                pbar.update(batch['image'].shape[0])

        cons_acc = self._evaluate(eval_loder)
        return cons_acc

    @torch.no_grad()
    def _evaluate(self, data_loader):
        coco = convert_to_coco_api(data_loader.dataset, bbox_fmt='coco')
        coco_evaluator = CocoEvaluator(coco, iou_types=["bbox"], bbox_fmt='coco')

        eval_net = get_hourglass(self.cfg.arch, num_classes=self.cfg.num_classes, is_training=False)
        if self.cfg.num_gpu > 1 and torch.cuda.is_available():
            eval_net = torch.nn.DataParallel(eval_net).cuda()
        else:
            eval_net = eval_net.to(self.cfg.device)
        eval_net.load_state_dict(self.model.state_dict())
        eval_net = eval_net.to(self.cfg.device)
        eval_net.eval()

        for inputs, targets in data_loader:
            targets = [{k: v.to(self.cfg.device) for k, v in t.items()} for t in targets]
            model_input = torch.stack(inputs, 0)
            model_input = model_input.to(self.cfg.device)
            output = eval_net(model_input)[-1]
            dets = ctdet_decode(*output, K=self.cfg.test_topk)
            # dets = dets.detach().cpu().numpy()
            res = {}
            for target, det in zip(targets, dets):
                labels = det[:, -1]
                scores = det[:, 4]
                boxes = det[:, :4]
                boxes[..., 2:] = boxes[..., 2:] - boxes[..., :2]  # Transform [x1, y1, x2, y2] to [x1, y1, w, h]
                boxes = boxes.reshape((boxes.shape[0], 1, 4))
                res[target["image_id"].item()] = {
                    "boxes": boxes,
                    "scores": scores,
                    "labels": labels,
                }
                coco_evaluator.update(res)

        coco_evaluator.synchronize_between_processes()
        coco_evaluator.accumulate()
        coco_evaluator.summarize()
        del eval_net
        return coco_evaluator