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
from src.model import get_model
from src.model.base.transformer import MultiHeadAttentionOne
from .optimizer import get_optimizer
from .dataset.dataset import get_val_loader, get_train_loader
from .util import intersectionAndUnionGPU, AverageMeter, get_model_dir_trans
from .test import validate_transformer
import argparse
from typing import Tuple
from .util import load_cfg_from_cfg_file, merge_cfg_from_list


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Training classifier weight transformer')
    parser.add_argument('--config', type=str, required=True, help='config file')
    parser.add_argument('--opts', default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()
    assert args.config is not None
    cfg = load_cfg_from_cfg_file(args.config)
    if args.opts is not None:
        cfg = merge_cfg_from_list(cfg, args.opts)
    return cfg


def main(args: argparse.Namespace) -> None:
    print(args)

    if args.manual_seed is not None:
        cudnn.benchmark = False  # 为True的话可以对网络结构固定、网络的输入形状不变的 模型提速
        cudnn.deterministic = True
        random.seed(args.manual_seed)
        np.random.seed(args.manual_seed)
        torch.manual_seed(args.manual_seed)
        torch.cuda.manual_seed(args.manual_seed)
        torch.cuda.manual_seed_all(args.manual_seed)

    # ====== Model + Optimizer ======
    model = get_model(args).cuda()

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
                        print('Mismatched shape {}: {}, {}'.format( key, pre_weight[key].shape, model_dict[key].shape))

            model.load_state_dict(model_dict, strict=True)
            print("=> loaded weight '{}'".format(fname))
        else:
            print("=> no weight found at '{}'".format(fname))

        # Fix the backbone layers
        for param in model.layer0.parameters():
            param.requires_grad = False
        for param in model.layer1.parameters():
            param.requires_grad = False
        for param in model.layer2.parameters():
            param.requires_grad = False
        for param in model.layer3.parameters():
            param.requires_grad = False
        for param in model.layer4.parameters():
            param.requires_grad = False
        for param in model.ppm.parameters():
            param.requires_grad = False
        for param in model.bottleneck.parameters():
            param.requires_grad = False

    # ====== Transformer ======
    trans_dim = args.bottleneck_dim

    transformer = MultiHeadAttentionOne(args.heads, trans_dim, trans_dim, trans_dim, dropout=0.5).cuda()

    optimizer_transformer = get_optimizer(args, [dict(params=transformer.parameters(), lr=args.trans_lr * args.scale_lr)])

    trans_save_dir = get_model_dir_trans(args)

    # ====== Data  ======
    train_loader, train_sampler = get_train_loader(args)
    episodic_val_loader, _ = get_val_loader(args)

    # ====== Metrics initialization ======
    max_val_mIoU = 0.
    if args.debug:
        iter_per_epoch = 5
    else:
        iter_per_epoch = args.iter_per_epoch if args.iter_per_epoch <= len(train_loader) else len(train_loader)

    log_iter = iter_per_epoch

    # ====== Training  ======
    print('==> Start training')
    for epoch in range(args.epochs):

        _, _ = do_epoch(
            args=args,
            train_loader=train_loader,
            iter_per_epoch=iter_per_epoch,
            model=model,
            transformer=transformer,
            optimizer_trans=optimizer_transformer,
            epoch=epoch,
            log_iter=log_iter,
        )

        val_Iou, val_loss = validate_transformer(
            args=args,
            val_loader=episodic_val_loader,
            model=model,
            transformer=transformer
        )

        # Model selection
        if val_Iou.item() > max_val_mIoU:
            max_val_mIoU = val_Iou.item()

            os.makedirs(trans_save_dir, exist_ok=True)
            filename_transformer = os.path.join(trans_save_dir, f'best.pth')

            if args.save_models:
                print('Saving checkpoint to: ' + filename_transformer)

                torch.save(
                    {'epoch': epoch,
                     'state_dict': transformer.state_dict(),
                     'optimizer': optimizer_transformer.state_dict()},
                    filename_transformer
                )

        print("=> Max_mIoU = {:.3f}".format(max_val_mIoU))

    if args.save_models:  # 所有跑完，存last epoch
        filename_transformer = os.path.join(trans_save_dir, 'final.pth')
        torch.save(
            {'epoch': args.epochs,
             'state_dict': transformer.state_dict(),
             'optimizer': optimizer_transformer.state_dict()},
            filename_transformer
        )


def do_epoch(
        args: argparse.Namespace,
        train_loader: torch.utils.data.DataLoader,
        model: nn.Module,
        transformer: nn.Module,
        optimizer_trans: torch.optim.Optimizer,
        epoch: int,
        iter_per_epoch: int,
        log_iter: int
) -> Tuple[torch.tensor, torch.tensor]:

    loss_meter = AverageMeter()
    train_losses = torch.zeros(log_iter)
    train_Ious = torch.zeros(log_iter)
    train_Ious0 = torch.zeros(log_iter)

    iterable_train_loader = iter(train_loader)

    model.eval()
    transformer.train()

    for i in range(iter_per_epoch):
        qry_img, q_label, spprt_imgs, s_label, subcls, _, _ = iterable_train_loader.next()

        if torch.cuda.is_available():
            spprt_imgs = spprt_imgs.cuda()  # [1, 1, 3, h, w]
            s_label = s_label.cuda()        # [1, 1, h, w]
            q_label = q_label.cuda()        # [1, h, w]
            qry_img = qry_img.cuda()        # [1, 3, h, w]

        # ====== Phase 1: Train the binary classifier on support samples ======

        # Keep the batch size/episode as 1.
        if spprt_imgs.shape[1] == 1:
            spprt_imgs_reshape = spprt_imgs.squeeze(0).expand(2, 3, args.image_size, args.image_size)                   # one shot 情况为什么要变为两个
            s_label_reshape = s_label.squeeze(0).expand(2, args.image_size, args.image_size).long()
        else:
            spprt_imgs_reshape = spprt_imgs.squeeze(0)  # [n_shots, 3, img_size, img_size]
            s_label_reshape = s_label.squeeze(0).long() # [n_shots, img_size, img_size]

        binary_cls = nn.Conv2d(args.bottleneck_dim, args.num_classes_tr, kernel_size=1, bias=False).cuda()  # classifier

        optimizer = optim.SGD(binary_cls.parameters(), lr=args.cls_lr)

        # Define loss function with Dynamic class weights
        s_label_arr = s_label.cpu().numpy().copy()  # [n_task, n_shots, img_size, img_size]
        back_pix = np.where(s_label_arr == 0)
        target_pix = np.where(s_label_arr == 1)

        criterion = nn.CrossEntropyLoss(
            weight=torch.tensor([1.0, len(back_pix[0]) / len(target_pix[0])]).cuda(),
            ignore_index=255)

        with torch.no_grad():
            f_s, _ = model.extract_features(spprt_imgs_reshape)  # [n_support, c, h, w]

        for index in range(args.adapt_iter):
            output_support = binary_cls(f_s)
            output_support = F.interpolate(
                output_support, size=s_label.size()[2:],
                mode='bilinear', align_corners=True
            )
            s_loss = criterion(output_support, s_label_reshape)
            optimizer.zero_grad()
            s_loss.backward()
            optimizer.step()

        # ====== Phase 2: Train the transformer to update the classifier's weights ======
        # Inputs of the transformer: weights of classifier trained on support sets, features of the query sample.

        # Dynamic class weights used for query image only during training
        q_label_arr = q_label.cpu().numpy().copy()  # [n_task, img_size, img_size]
        q_back_pix = np.where(q_label_arr == 0)
        q_target_pix = np.where(q_label_arr == 1)

        criterion = nn.CrossEntropyLoss(
            weight=torch.tensor([1.0, len(q_back_pix[0]) / (len(q_target_pix[0]) + 1e-12)]).cuda(),
            ignore_index=255)

        model.eval()
        with torch.no_grad():
            f_q, _ = model.extract_features(qry_img)  # [1, c, h, w]
            pred_q0 = binary_cls(f_q)
            pred_q0 = F.interpolate(pred_q0, size=q_label.shape[1:], mode='bilinear', align_corners=True)
            f_q = F.normalize(f_q, dim=1)          # [1, c, 60, 60]

        # Weights of the classifier.
        weights_cls = binary_cls.weight.data       # [2, 512, 1, 1]
        weights_cls_reshape = weights_cls.squeeze().unsqueeze(0).expand(args.batch_size, 2, weights_cls.shape[1])  # [B, 2, c]

        # Update the classifier's weights with transformer
        updated_weights_cls = transformer(weights_cls_reshape, f_q, f_q)  # [n_task, 2, c] [1, 2, 512]

        f_q_reshape = f_q.view(args.batch_size, args.bottleneck_dim, -1)  # [n_task, c, hw]

        pred_q = torch.matmul(updated_weights_cls, f_q_reshape).view(args.batch_size, 2, f_q.shape[-2], f_q.shape[-1])             # [n_task, 2, h, w]
        pred_q = F.interpolate(pred_q, size=q_label.shape[1:],mode='bilinear', align_corners=True)

        loss_q = criterion(pred_q, q_label.long())
        optimizer_trans.zero_grad()
        loss_q.backward()
        optimizer_trans.step()

        # Print loss and mIoU
        intersection, union, target = intersectionAndUnionGPU(pred_q.argmax(1), q_label, args.num_classes_tr, 255)
        IoUb, IoUf = intersection / (union + 1e-10)
        loss_meter.update(loss_q.item() / args.batch_size)

        train_losses[i] = loss_meter.avg
        train_Ious[i] = (IoUb + IoUf)/2

        intersection, union, target = intersectionAndUnionGPU(pred_q0.argmax(1), q_label, args.num_classes_tr, 255)
        IoUb0, IoUf0 = intersection / (union + 1e-10)
        train_Ious0[i] = (IoUb0 + IoUf0)/2

        if (epoch == 0 and i%100==0) or i%500==0:
            print('iter {} IoUf {:.2f}, IoUb {:.2f}, IoUf0 {:.2f}, IoUb0 {:.2f}, pred_q0 {}'.format(
                i, IoUf, IoUb, IoUf0, IoUb0, torch.bincount(pred_q0.argmax(1).view(-1)).cpu().numpy()
            ))
    print('Epoch {}: The mIoU {:.2f}, loss {:.2f}, mIoU0 {:.2f}'.format(
        epoch + 1, train_Ious.mean(), train_losses.mean(), train_Ious0.mean() ))

    return train_Ious, train_losses


if __name__ == "__main__":
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(str(x) for x in args.gpus)

    if args.debug:
        args.test_num = 500
        args.epochs = 2
        args.n_runs = 2
        args.save_models = False

    world_size = len(args.gpus)
    args.distributed = (world_size > 1)
    main(args)