# encoding:utf-8

import os
import time
import random
import torch.backends.cudnn as cudnn
import torch.nn.parallel
import torch.utils.data
from collections import defaultdict
from .model import *
from .optimizer import get_optimizer, get_scheduler
from .dataset.dataset import get_val_loader, get_train_loader
from .util import intersectionAndUnionGPU, AverageMeter, CompareMeter
from .util import load_cfg_from_cfg_file, merge_cfg_from_list, ensure_path, set_log_path, log
import argparse


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

    sv_path = 'fuse_{}/{}{}/split{}_shot{}/{}'.format(
        args.train_name, args.arch, args.layers, args.train_split, args.shot, args.exp_name)
    sv_path = os.path.join('./results', sv_path)
    ensure_path(sv_path)
    set_log_path(path=sv_path)
    log('save_path {}'.format(sv_path))

    log(args)

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
        fname = args.resume_weights + args.train_name + '/' + \
                'split={}/pspnet_{}{}/best.pth'.format(args.train_split, args.arch, args.layers)
        if os.path.isfile(fname):
            log("=> loading weight '{}'".format(fname))
            pre_weight = torch.load(fname)['state_dict']
            pre_dict = model.state_dict()

            for index, key in enumerate(pre_dict.keys()):
                if 'classifier' not in key and 'gamma' not in key:
                    if pre_dict[key].shape == pre_weight['module.' + key].shape:
                        pre_dict[key] = pre_weight['module.' + key]
                    else:
                        log('Pre-trained shape and model shape for {}: {}, {}'.format(
                            key, pre_weight['module.' + key].shape, pre_dict[key].shape))
                        continue

            model.load_state_dict(pre_dict, strict=True)
            log("=> loaded weight '{}'".format(fname))
        else:
            log("=> no weight found at '{}'".format(fname))

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

    # ========= Data  ==========
    train_loader, train_sampler = get_train_loader(args)
    episodic_val_loader, _ = get_val_loader(args)

    # ======= Transformer =======
    if args.crm_type == 'chm':
        FusionNet = CHMLearner(ktype='psi', feat_dim=1024, temp=args.temp).cuda()
        print('model setting kernel type {}, input feature dim {}'.format('psi', 2048))
    else:
        FusionNet = MatchNet(temp=args.temp, cv_type='red', sce=args.sce, cyc=args.cyc, sym_mode=True).cuda()
    optimizer_meta = get_optimizer(args, [dict(params=FusionNet.parameters(), lr=args.trans_lr * args.scale_lr)])
    scheduler = get_scheduler(args, optimizer_meta, len(train_loader))

    # ====== Metrics initialization ======
    max_val_mIoU, max_val_mIoU1 = 0., 0.

    # ====== Training  ======
    log('==> Start training')
    for epoch in range(1, args.epochs+1):

        train_loss_meter1 = AverageMeter()
        train_iou_meter1 = AverageMeter()
        train_loss_meter0 = AverageMeter()
        train_iou_meter0 = AverageMeter()
        train_iou_compare = CompareMeter()

        iterable_train_loader = iter(train_loader)
        for i in range(1, len(train_loader)+1):
            qry_img, q_label, spt_imgs, s_label, subcls, _, _ = iterable_train_loader.next()  # q: [1, 3, 473, 473], s: [1, 1, 3, 473, 473]

            if torch.cuda.is_available():
                spt_imgs = spt_imgs.cuda()  # [1, 1, 3, h, w]
                s_label = s_label.cuda()  # [1, 1, h, w]
                q_label = q_label.cuda()  # [1, h, w]
                qry_img = qry_img.cuda()  # [1, 3, h, w]

            # ====== Phase 1: Train the binary classifier on support samples ======

            spt_imgs = spt_imgs.squeeze(0)       # [n_shots, 3, img_size, img_size]
            s_label = s_label.squeeze(0).long()  # [n_shots, img_size, img_size]

            # fine-tune classifier
            model.eval()
            with torch.no_grad():
                f_s, fs_lst = model.extract_features(spt_imgs)  # f_s为ppm之后的feat, fs_lst为mid_feat
            model.inner_loop(f_s, s_label)

            # ====== Phase 2: Train the attention to update query score  ======
            model.eval()
            with torch.no_grad():
                f_q, fq_lst = model.extract_features(qry_img)  # [n_task, c, h, w]
                pd_q0 = model.classifier(f_q)
                pd_s  = model.classifier(f_s)  # [B, 2, 60, 60]
                pred_q0 = F.interpolate(pd_q0, size=q_label.shape[1:], mode='bilinear', align_corners=True)

            # filter out ignore pixels
            if args.rmid == 'nr':
                idx = -1
            elif 'mid' in args.rmid or 'l' in args.rmid:
                idx = int(args.rmid[-1]) - 2
            fs_fea = fs_lst[idx]  # [1, 2048, 60, 60]
            fq_fea = fq_lst[idx]  # [1, 2048, 60, 60]

            if args.crm_type == 'chm':
                fs_fea = F.interpolate(fs_fea, scale_factor=0.5, mode='bilinear', align_corners=True)
                fq_fea = F.interpolate(fq_fea, scale_factor=0.5, mode='bilinear', align_corners=True)
                weighted_v = FusionNet(fq_fea, fs_fea, v=f_s.view(f_s.shape[:2] +(-1,)), ig_mask=None, ret_corr=False)
            else:
                use_cyc = False  # (i >= 1000) and args.cyc
                weighted_v = FusionNet(fq_fea, fs_fea, v=f_s.view(f_s.shape[:2] +(-1,)), s_mask=pd_s.argmax(1), use_cyc=use_cyc, ig_mask=None)
            pd_q1 = model.classifier(weighted_v)
            pred_q1 = F.interpolate(pd_q1, size=q_label.shape[-2:], mode='bilinear', align_corners=True)

            out = (weighted_v * args.att_wt + f_q) / (1 + args.att_wt)
            pd_q = model.classifier(out)
            pred_q = F.interpolate(pd_q, size=q_label.shape[-2:], mode='bilinear', align_corners=True)

            # Loss function: Dynamic class weights used for query image only during training
            q_label_arr = q_label.cpu().numpy().copy()  # [n_task, img_size, img_size]
            q_back_pix = np.where(q_label_arr == 0)
            q_target_pix = np.where(q_label_arr == 1)
            loss_weight = torch.tensor([1.0, len(q_back_pix[0]) / (len(q_target_pix[0]) + 1e-12)]).cuda()
            criterion = nn.CrossEntropyLoss(weight=loss_weight, ignore_index=255)
            q_loss1 = criterion(pred_q1, q_label.long())
            q_loss0 = criterion(pred_q0, q_label.long())

            if args.aux:
                wt_mask = ((pred_q0.argmax(dim=1) != pred_q1.argmax(dim=1)) & (q_label != 255)).float()
                aux_criterion = torch.nn.CrossEntropyLoss(ignore_index=255, reduction='none')
                wt_mask[wt_mask == 0.0] = 0.001
                aux_loss = 2.0 * torch.sum(aux_criterion(pred_q, q_label.long()) * wt_mask) / torch.sum(wt_mask)  # assume batch_size 1

                q_loss1 = q_loss1 + aux_loss
            else: aux_loss=0.0

            optimizer_meta.zero_grad()
            q_loss1.backward()
            optimizer_meta.step()
            if args.scheduler == 'cosine':
                scheduler.step()

            # Print loss and mIoU
            IoUb, IoUf = dict(), dict()
            for (pred, idx) in [(pred_q0, 0), (pred_q1, 1), (pred_q, 2)]:
                intersection, union, target = intersectionAndUnionGPU(pred.argmax(1), q_label, args.num_classes_tr, 255)
                IoUb[idx], IoUf[idx] = (intersection / (union + 1e-10)).cpu().numpy()  # mean of BG and FG

            train_loss_meter0.update(q_loss0.item() / args.batch_size, 1)
            train_iou_meter0.update((IoUf[0]+IoUb[0])/2, 1)
            train_loss_meter1.update(q_loss1.item() / args.batch_size, 1)
            train_iou_meter1.update((IoUf[1] + IoUb[1]) / 2, 1)
            train_iou_compare.update(IoUf[1], IoUf[0])

            if i%100==0 or (epoch==1 and i<=1190):
                log('Ep{}/{} IoUf0 {:.2f} IoUb0 {:.2f} IoUf1 {:.2f} IoUb1 {:.2f} IoUf {:.2f} IoUb {:.2f} '
                    'loss0 {:.2f} loss1 {:.2f} auxLoss {:.2f} lr {:.4f}'.format(
                    epoch, i, IoUf[0], IoUb[0], IoUf[1], IoUb[1], IoUf[2], IoUb[2],
                    q_loss0, q_loss1, aux_loss, optimizer_meta.param_groups[0]['lr']))
            if i%1190==0:
                log('------Ep{}/{} FG IoU1 compared to IoU0 win {}/{} avg diff {:.2f}'.format(epoch, i,
                    train_iou_compare.win_cnt, train_iou_compare.cnt, train_iou_compare.diff_avg))
                train_iou_compare.reset()
                val_Iou, val_Iou1, val_loss = validate_epoch(args=args, val_loader=episodic_val_loader, model=model, Net=FusionNet)

                # Model selection
                if val_Iou.item() > max_val_mIoU:
                    max_val_mIoU = val_Iou.item()
                    filename_transformer = os.path.join(sv_path, f'best.pth')
                    if args.save_models:
                        log('=> Max_mIoU = {:.3f} Saving checkpoint to: {}'.format(max_val_mIoU, filename_transformer))
                        torch.save({'epoch': epoch, 'state_dict': FusionNet.state_dict(),
                                    'optimizer': optimizer_meta.state_dict()}, filename_transformer)
                if val_Iou1.item() > max_val_mIoU1:
                    max_val_mIoU1 = val_Iou1.item()
                    filename_transformer = os.path.join(sv_path, f'best1.pth')
                    if args.save_models:
                        log('=> Max_mIoU1 = {:.3f} Saving checkpoint to: {}'.format(max_val_mIoU1, filename_transformer))
                        torch.save({'epoch': epoch, 'state_dict': FusionNet.state_dict(),
                                    'optimizer': optimizer_meta.state_dict()}, filename_transformer)

        log('===========Epoch {}===========: The mIoU0 {:.2f}, mIoU1 {:.2f}, loss0 {:.2f}, loss1 {:.2f}===========\n'.format(
            epoch, train_iou_meter0.avg, train_iou_meter1.avg, train_loss_meter0.avg, train_loss_meter1.avg))
        train_iou_meter1.reset()
        train_loss_meter1.reset()

    if args.save_models:  # 所有跑完，存last epoch
        filename_transformer = os.path.join(sv_path, 'final.pth')
        torch.save(
            {'epoch': args.epochs,
             'state_dict': FusionNet.state_dict(),
             'optimizer': optimizer_meta.state_dict()},
            filename_transformer)


def validate_epoch(args, val_loader, model, Net):
    log('==> Start testing')

    iter_num = 0
    start_time = time.time()
    loss_meter = AverageMeter()

    cls_intersection = defaultdict(int)  # Default value is 0
    cls_union = defaultdict(int)
    IoU = defaultdict(float)

    cls_intersection0 = defaultdict(int)  # Default value is 0
    cls_union0 = defaultdict(int)
    IoU0 = defaultdict(float)

    cls_intersection1 = defaultdict(int)  # Default value is 0
    cls_union1 = defaultdict(int)
    IoU1 = defaultdict(float)

    cls_intersection1b = defaultdict(int)  # Default value is 0
    cls_union1b = defaultdict(int)
    IoU1b = defaultdict(float)

    val_iou_compare = CompareMeter()

    for e in range(args.test_num):

        iter_num += 1
        try:
            qry_img, q_label, spt_imgs, s_label, subcls, spprt_oris, qry_oris = iter_loader.next()
        except:
            iter_loader = iter(val_loader)
            qry_img, q_label, spt_imgs, s_label, subcls, spprt_oris, qry_oris = iter_loader.next()
        if torch.cuda.is_available():
            spt_imgs = spt_imgs.cuda()
            s_label = s_label.cuda()
            q_label = q_label.cuda()
            qry_img = qry_img.cuda()

        # ====== Phase 1: Train a new binary classifier on support samples. ======
        spt_imgs = spt_imgs.squeeze(0)   # [n_shots, 3, img_size, img_size]
        s_label = s_label.squeeze(0).long()  # [n_shots, img_size, img_size]

        # fine-tune classifier
        model.eval()
        with torch.no_grad():
            f_s, fs_lst = model.extract_features(spt_imgs)
        model.inner_loop(f_s, s_label)

        # ====== Phase 2: Update query score using attention. ======
        model.eval()
        with torch.no_grad():
            f_q, fq_lst = model.extract_features(qry_img)  # [n_task, c, h, w]
            pd_q0 = model.classifier(f_q)
            pd_s  = model.classifier(f_s)
            pred_q0 = F.interpolate(pd_q0, size=q_label.shape[1:], mode='bilinear', align_corners=True)

            if args.rmid == 'nr':
                idx = -1
            elif args.rmid in ['mid2', 'mid3', 'mid4']:
                idx = int(args.rmid[-1]) - 2
            fs_fea = fs_lst[idx]  # [1, 2048, 60, 60]
            fq_fea = fq_lst[idx]  # [1, 2048, 60, 60]

            if args.crm_type == 'chm':
                fs_fea = F.interpolate(fs_fea, scale_factor=0.5, mode='bilinear', align_corners=True)
                fq_fea = F.interpolate(fq_fea, scale_factor=0.5, mode='bilinear', align_corners=True)
                weighted_v = Net(fq_fea, fs_fea, v=f_s.view(f_s.shape[:2] + (-1,)), ig_mask=None, ret_corr=False)
            else:
                weighted_v, corr1 = Net(fq_fea, fs_fea, v=f_s.view(f_s.shape[:2] + (-1,)), s_mask=pd_s.argmax(1), use_cyc=args.cyc, ig_mask=None, ret_corr=True)

            if args.ignore:
                ig_mask = get_ig_mask(sim=corr1, s_label=s_label, q_label=q_label, pd_q0=pd_q0, pd_s=pd_s)   # s_label & q_label 为原图大小
                weighted_v = att_weighted_out(sim=corr1, v=f_q, temp=args.temp, ig_mask=ig_mask)

            pd_q1 = model.classifier(weighted_v)
            pred_q1 = F.interpolate(pd_q1, size=q_label.shape[-2:], mode='bilinear', align_corners=True)
            out = (weighted_v * args.att_wt + f_q) / (1 + args.att_wt)
            pd_q = model.classifier(out)
            pred_q = F.interpolate(pd_q, size=q_label.shape[-2:], mode='bilinear', align_corners=True)

        # IoU and loss
        curr_cls = subcls[0].item()  # 当前episode所关注的cls
        for id, (cls_intersection_, cls_union_, IoU_, pred) in \
                enumerate( [(cls_intersection0, cls_union0, IoU0, pred_q0), (cls_intersection1, cls_union1, IoU1, pred_q1),
                 (cls_intersection, cls_union, IoU, pred_q),] ):
            intersection, union, target = intersectionAndUnionGPU(pred.argmax(1), q_label, 2, 255)
            intersection, union = intersection.cpu(), union.cpu()
            cls_intersection_[curr_cls] += intersection[1]  # only consider the FG
            cls_union_[curr_cls] += union[1]                # only consider the FG
            IoU_[curr_cls] = cls_intersection_[curr_cls] / (cls_union_[curr_cls] + 1e-10)   # cls wise IoU
            if id==0: iouf0 = intersection[1]/union[1]     # fg IoU for the current episode
            elif id==1: iouf1 = intersection[1]/union[1]
        val_iou_compare.update(iouf1,iouf0)   # compare 当前episode的IoU of att pred and pred0

        criterion_standard = nn.CrossEntropyLoss(ignore_index=255)
        loss1 = criterion_standard(pred_q1, q_label)
        loss_meter.update(loss1.item())

        if (iter_num % 200 == 0):
            mIoU = np.mean([IoU[i] for i in IoU])                                  # mIoU across cls
            mIoU0 = np.mean([IoU0[i] for i in IoU0])
            mIoU1 = np.mean([IoU1[i] for i in IoU1])

            log('Test: [{}/{}] mIoU0 {:.4f} mIoU1 {:.4f} mIoU {:.4f} Loss {loss_meter.val:.4f} ({loss_meter.avg:.4f}) '.format(
                iter_num, args.test_num, mIoU0, mIoU1, mIoU, loss_meter=loss_meter))

    runtime = time.time() - start_time
    mIoU = np.mean(list(IoU.values()))  # IoU: dict{cls: cls-wise IoU}
    log('mIoU---Val result: mIoU0 {:.4f}, mIoU1 {:.4f} mIoU {:.4f} | time used {:.1f}m.'.format(
        mIoU0, mIoU1, mIoU, runtime/60))
    for class_ in cls_union:
        log("Class {} : {:.4f}".format(class_, IoU[class_]))
    log('------Val FG IoU1 compared to IoU0 win {}/{} avg diff {:.2f}'.format(
        val_iou_compare.win_cnt, val_iou_compare.cnt, val_iou_compare.diff_avg))

    return mIoU, mIoU1, loss_meter.avg


if __name__ == "__main__":
    args = parse_args()

    main(args)