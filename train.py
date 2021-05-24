import os
import sys
import time
import argparse
from easydict import EasyDict as edict
import torch.utils.data

from CenterNet import CenterNet
from tools.Knee_dataset import KneeDataset, collate_fn
from config import cfg as Cfg
from utils.summary import create_summary, create_logger

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'lib'))

# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
# os.environ["CUDA_VISIBLE_DEVICES"] = '0'


# Training settings
def get_args(**kwargs):
    cfg = kwargs
    parser = argparse.ArgumentParser(description='simple_centernet45')

    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--root_dir', type=str, default='./')
    parser.add_argument('--data_dir', type=str, default='./data')
    parser.add_argument('--log_name', type=str, default='test')
    parser.add_argument('--pretrain_name', type=str, default='pretrain')

    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--num_epochs', type=int, default=250)
    parser.add_argument('--test_topk', type=int, default=100)

    parser.add_argument('--log_interval', type=int, default=100)
    parser.add_argument('--val_interval', type=int, default=5)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--num_classes', type=int, default=4)

    args = vars(parser.parse_args())
    cfg.update(args)

    return edict(cfg)


def train():
    cfg = get_args(**Cfg)
    os.chdir(cfg.root_dir)
    os.makedirs(cfg.log_dir, exist_ok=True)
    os.makedirs(cfg.ckpt_dir, exist_ok=True)

    logger = create_logger(cfg.local_rank, save_dir=cfg.log_dir)
    print = logger.info
    print(cfg)

    torch.manual_seed(317)
    torch.backends.cudnn.benchmark = True

    cfg.device = torch.device('cuda')

    print('Setting up data...')
    train_dataset = KneeDataset(cfg.train_label, cfg)
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=cfg.batch_size,
                                               shuffle=not cfg.dist,
                                               num_workers=cfg.num_workers,
                                               pin_memory=True,
                                               drop_last=True)

    val_dataset = KneeDataset(cfg.val_label, cfg, False)
    eval_loader = torch.utils.data.DataLoader(val_dataset, batch_size=cfg.batch_size,
                                             shuffle=False, num_workers=1, pin_memory=True,
                                             collate_fn= collate_fn)
    center_net = CenterNet(cfg)
    center_net.train(len(train_dataset), train_loader, eval_loader, cfg.num_epochs)


if __name__ == '__main__':
    train()
