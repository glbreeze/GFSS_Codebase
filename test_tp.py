# encoding:utf-8

import os
import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.utils.data
import torch.optim as optim
from collections import defaultdict
from src.dataset.dataset import get_val_loader
from src.util import AverageMeter, batch_intersectionAndUnionGPU, get_model_dir_trans
from src.model import get_model, CosCls
from src.model.base.transformer import MultiHeadAttentionOne
from src.util import load_cfg_from_cfg_file, merge_cfg_from_list, log
import argparse
from torch.nn.parallel import DistributedDataParallel as DDP
import time
from typing import Tuple

# =================== get config ===================

arg_input = ' --config config_files/coco.yaml   \
  --opts train_split 0   layers 50    shot 1   trans_lr 0.001   cls_lr 0.1    batch_size 1  \
  batch_size_val 1   epochs 20     test_num 1000 '

parser = argparse.ArgumentParser(description='Training classifier weight transformer')
parser.add_argument('--config', type=str, required=True, help='config_files/pascal.yaml')
parser.add_argument('--opts', default=None, nargs=argparse.REMAINDER)
args = parser.parse_args(arg_input.split())

assert args.config is not None
cfg = load_cfg_from_cfg_file(args.config)
if args.opts is not None:
    cfg = merge_cfg_from_list(cfg, args.opts)
args = cfg
print(args)


# =================== load model ===================
model = get_model(args)

trans_dim = args.bottleneck_dim
transformer = MultiHeadAttentionOne(args.heads, trans_dim, trans_dim, trans_dim, dropout=0.5)

root_trans = get_model_dir_trans(args)  # root for transformer

args.wt_file = 0
if args.resume_weights:
    if args.get('wt_file', 0) == 1:
        fname = args.resume_weights + args.train_name + '/' + 'split={}/pspnet_{}{}/best1.pth'.format(args.train_split,
                                                                                                      args.arch,
                                                                                                      args.layers)
    else:
        fname = args.resume_weights + args.train_name + '/' + 'split={}/pspnet_{}{}/best.pth'.format(args.train_split,
                                                                                                     args.arch,
                                                                                                     args.layers)
    if os.path.isfile(fname):
        print("=> loading weight '{}'".format(fname))
        pre_weight = torch.load(fname)['state_dict']
        model_dict = model.state_dict()
        for index, key in enumerate(model_dict.keys()):
            if 'classifier' not in key and 'gamma' not in key:
                if model_dict[key].shape == pre_weight[key].shape:
                    model_dict[key] = pre_weight[key]
                else:
                    print('Mismatched shape {}: {}, {}'.format(key, pre_weight[key].shape, model_dict[key].shape))

        model.load_state_dict(model_dict, strict=True)
        print("=> loaded weight '{}'".format(fname))
    else:
        print("=> no weight found at '{}'".format(fname))

if args.ckpt_used is not None:
    filepath = os.path.join(root_trans, f'{args.ckpt_used}.pth')
    assert os.path.isfile(filepath), filepath
    print("=> loading transformer weight '{}'".format(filepath))
    checkpoint = torch.load(filepath)
    transformer.load_state_dict(checkpoint['state_dict'])
    print("=> loaded transformer weight '{}'".format(filepath))
else:
    print("=> Not loading anything")

# ====== Data  ======
args.train_name = 'pascal'
args.train_list = 'lists/pascal/train.txt'
args.data_root = '../dataset/VOCdevkit/VOC2012'
args.val_list = 'lists/pascal/val.txt'
episodic_val_loader, _ = get_val_loader(args)

# ====== Test  ======
model.eval()
transformer.eval()
nb_episodes = int(args.test_num / args.batch_size_val)

H, W = args.image_size, args.image_size
if args.image_size == 473:
    h, w = 60, 60
else:
    h, w = model.feature_res  # (53, 53)


# ====== Initialize the metric dictionaries ======
loss_meter = AverageMeter()
iter_num, runtime = 0, 0
cls_intersection_base = defaultdict(int)  # Default value is 0
cls_union_base = defaultdict(int)
cls_intersection_cwt = defaultdict(int)  # Default value is 0
cls_union_cwt = defaultdict(int)
cls_intersection_base1 = defaultdict(int)  # Default value is 0
cls_union_base1 = defaultdict(int)
cls_intersection_cwt1 = defaultdict(int)  # Default value is 0
cls_union_cwt1 = defaultdict(int)

IoU_base = defaultdict(int)
IoU_cwt = defaultdict(int)
IoU_base1 = defaultdict(int)
IoU_cwt1 = defaultdict(int)

# ================= test model ====================
val_loader=episodic_val_loader
iter_loader = iter(val_loader)
qry_img, q_label, spprt_imgs, s_label, subcls, spprt_oris, qry_oris = iter_loader.next()
spprt_imgs, s_label = spprt_imgs.squeeze(0), s_label.squeeze(0)  # [2, 3, 473, 473], [2, 473, 473]


# ====== Phase 1: Train a new binary classifier on support samples. ======

def finetune_cls(spt_imgs, spt_labels, model):

    binary_classifier = nn.Conv2d(args.bottleneck_dim, args.num_classes_tr, kernel_size=1, bias=False)
    optimizer = optim.SGD(binary_classifier.parameters(), lr=args.cls_lr)

    # Dynamic class weights
    s_label_arr = spt_labels.cpu().numpy().copy()  # [n_task, n_shots, img_size, img_size]
    back_pix = np.where(s_label_arr == 0)
    target_pix = np.where(s_label_arr == 1)

    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor([1.0, len(back_pix[0]) / len(target_pix[0])]),
        ignore_index=255)

    with torch.no_grad():
        f_s, _ = model.extract_features(spt_imgs)  # [ n_shots, c, h, w]

    for index in range(args.adapt_iter):
        output_support = binary_classifier(f_s)
        output_support = F.interpolate(output_support, size=spt_labels.size()[-2:], mode='bilinear', align_corners=True)
        s_loss = criterion(output_support, spt_labels)
        optimizer.zero_grad()
        s_loss.backward()
        optimizer.step()

    return binary_classifier

binary_classifier0 = finetune_cls(spprt_imgs[0:1], s_label[0:1], model)  # only using ori image
binary_classifier1 = finetune_cls(spprt_imgs, s_label, model)

# ====== Phase 2: Update classifier's weights with old weights and query features. ======
pred_base = []
pred_cwt = []
for binary_classifier in [binary_classifier0, binary_classifier1]:
    with torch.no_grad():
        f_q, _ = model.extract_features(qry_img)  # [n_task, c, h, w]
        pred_q0 = binary_classifier(f_q)
        pred_base.append(pred_q0)

        f_q = F.normalize(f_q, dim=1)
        weights_cls = binary_classifier.weight.data  # [2, c, 1, 1]
        weights_cls_reshape = weights_cls.squeeze().unsqueeze(0).expand(f_q.shape[0], 2, 512)  # [1, 2, c]

        print('=====>weight cls {}, f_q {}, transformer {}'.format(weights_cls.device, f_q.device, next(transformer.parameters()).device))

        updated_weights_cls = transformer(weights_cls_reshape, f_q, f_q)  # [1, 2, c]

        # Build a temporary new classifier for prediction
        Pseudo_cls = nn.Conv2d(args.bottleneck_dim, args.num_classes_tr, kernel_size=1, bias=False)
        # Initialize the weights with updated ones
        Pseudo_cls.weight.data = torch.as_tensor(updated_weights_cls.squeeze(0).unsqueeze(2).unsqueeze(3))
        pred_q = Pseudo_cls(f_q)   # [1, 2, 60, 60] 没有expand到2个
        pred_cwt.append(pred_q)

logits_base[i] = pred_base[0].detach()  # finetune pred for ori support set
logits_base1[i] = pred_base[1].detach()  # finetune pred for augmented support set
logits_cwt[i] = pred_cwt[0].detach()     # [1 batch_size, 2 channel, 60, 60] 其实一个batch只有一个obs, i=0
logits_cwt1[i] = pred_cwt[1].detach()    # cwt pred for augmented




