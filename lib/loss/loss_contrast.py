from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from abc import ABC

import torch
import torch.nn as nn
import torch.nn.functional as F

from lib.loss.loss_helper import FSAuxCELoss, FSAuxRMILoss, FSCELoss
from lib.utils.tools.logger import Logger as Log


class PixelContrastLoss(nn.Module, ABC):
    def __init__(self, configer):
        super(PixelContrastLoss, self).__init__()

        self.configer = configer
        self.bg_anchor = configer.get('contrast', 'bg_anchor')
        self.temperature = self.configer.get('contrast', 'temperature')
        self.base_temperature = self.configer.get('contrast', 'base_temperature')

        self.max_samples = self.configer.get('contrast', 'max_samples')
        self.max_views = self.configer.get('contrast', 'max_views')
        self.ignore_label = 255

    def _hard_anchor_sampling(self, X, y_hat, y):  # x: embed, y: pred, y_hat: GT
        batch_size, feat_dim = X.shape[0], X.shape[-1]

        classes = []
        total_classes = 0
        for ii in range(batch_size):
            this_y = y_hat[ii]
            this_classes = torch.unique(this_y)
            this_classes = [x for x in this_classes if x != self.ignore_label]
            this_classes = [x for x in this_classes if (this_y == x).nonzero().shape[0] > self.max_views]

            classes.append(this_classes)   # list of list [[],[]]
            total_classes += len(this_classes)

        if total_classes == 0:
            return None, None

        # max_samples: max num pixels/anchors in each batch, total_classes: img * num_of_classes in the img
        n_view = self.max_samples // total_classes
        n_view = min(n_view, self.max_views)

        X_ = torch.zeros((total_classes, n_view, feat_dim), dtype=torch.float).to(X.device)  # 所选的anchor point的embed
        y_ = torch.zeros(total_classes, dtype=torch.float).to(y.device)                      # 所选的anchor point的label

        X_ptr = 0
        for ii in range(batch_size):
            this_y_hat = y_hat[ii]      # [3600]
            this_y = y[ii]              # [3600]
            this_classes = classes[ii]

            for cls_id in this_classes:
                hard_indices = ((this_y_hat == cls_id) & (this_y != cls_id)).nonzero()
                easy_indices = ((this_y_hat == cls_id) & (this_y == cls_id)).nonzero()

                num_hard = hard_indices.shape[0]
                num_easy = easy_indices.shape[0]

                if num_hard >= n_view / 2 and num_easy >= n_view / 2:
                    num_hard_keep = n_view // 2
                    num_easy_keep = n_view - num_hard_keep
                elif num_hard >= n_view / 2:
                    num_easy_keep = num_easy
                    num_hard_keep = n_view - num_easy_keep
                elif num_easy >= n_view / 2:
                    num_hard_keep = num_hard
                    num_easy_keep = n_view - num_hard_keep
                else:
                    print('this shoud be never touched! {} {} {}'.format(num_hard, num_easy, n_view))
                    raise Exception

                perm = torch.randperm(num_hard)
                hard_indices = hard_indices[perm[:num_hard_keep]]
                perm = torch.randperm(num_easy)
                easy_indices = easy_indices[perm[:num_easy_keep]]
                indices = torch.cat((hard_indices, easy_indices), dim=0)

                X_[X_ptr, :, :] = X[ii, indices, :].squeeze(1)
                y_[X_ptr] = cls_id
                X_ptr += 1

        return X_, y_

    def _contrastive(self, feats_, labels_):
        anchor_num, n_view = feats_.shape[0], feats_.shape[1]

        contrast_count = n_view
        contrast_feature = torch.cat(torch.unbind(feats_, dim=1), dim=0)  # [17, 60, 256] -> (17, 256) * 60  [1020, 256]

        anchor_feature = contrast_feature
        anchor_count = contrast_count   # 100 every img every class

        anchor_dot_contrast = torch.div(torch.matmul(anchor_feature, torch.transpose(contrast_feature, 0, 1)), self.temperature)
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)   # [1020, 1]
        logits = anchor_dot_contrast - logits_max.detach()                    # [1020, 1020]

        labels_ = labels_.contiguous().view(-1, 1)
        mask = torch.eq(labels_, torch.transpose(labels_, 0, 1)).float().to(labels_.device)
        mask = mask.repeat(anchor_count, contrast_count)
        neg_mask = 1 - mask
        logits_mask = torch.ones_like(mask).scatter_(1, torch.arange(anchor_num * anchor_count).view(-1, 1).to(mask.device), 0)
        mask = mask * logits_mask    # mask of anchor*pos samples pairs (logits_mask gets rid of self)

        neg_logits = torch.exp(logits) * neg_mask          # 当前 pixel 不同的 负类 (每一行中）
        neg_logits = neg_logits.sum(1, keepdim=True)       # [1020, 1], sum of exp(negative_sample) for all 1020 anchors

        exp_logits = torch.exp(logits)                              # [1020, 1020]
        log_prob = logits - torch.log(exp_logits + neg_logits)      # [1020, 1020]
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)  # [1020]

        if not self.bg_anchor:
            non_bg_mask = (labels_.view(-1)!=0).repeat(anchor_count)  # [1020]
            mean_log_prob_pos = mean_log_prob_pos[non_bg_mask]        # NCE loss for non BG anchor pixels

        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.mean()

        return loss

    def forward(self, feats, labels=None, predict=None):  # feats : [8, 256, 60, 60], labels: [8, 473, 473], predict: [8, 60, 60]
        # align shape of the labels with feats (embedding)
        labels = labels.unsqueeze(1).float().clone()
        labels = torch.nn.functional.interpolate(labels, (feats.shape[2], feats.shape[3]), mode='nearest')
        labels = labels.squeeze(1).long()
        assert labels.shape[-1] == feats.shape[-1], '{} {}'.format(labels.shape, feats.shape)

        batch_size = feats.shape[0]

        labels = labels.contiguous().view(batch_size, -1)                     # [8, 3600]
        predict = predict.contiguous().view(batch_size, -1)                   # [8, 3600]
        feats = feats.permute(0, 2, 3, 1)
        feats = feats.contiguous().view(feats.shape[0], -1, feats.shape[-1])  # [8, 3600, 256]

        feats_, labels_ = self._hard_anchor_sampling(feats, labels, predict)  # feats_: [17, 60, 256], labels: [17]

        loss = self._contrastive(feats_, labels_)
        return loss


class ContrastCELoss(nn.Module, ABC):
    def __init__(self, configer=None):
        super(ContrastCELoss, self).__init__()
        self.configer = configer

        ignore_index = 255
        if self.configer.exists('loss', 'params') and 'ce_ignore_index' in self.configer.get('loss', 'params'):
            ignore_index = self.configer.get('loss', 'params')['ce_ignore_index']
        Log.info('ignore_index: {}'.format(ignore_index))

        self.loss_weight = self.configer.get('contrast', 'loss_weight')
        self.use_rmi = self.configer.get('contrast', 'use_rmi')

        if self.use_rmi:
            self.seg_criterion = FSAuxRMILoss(configer=configer)
        else:
            self.seg_criterion = FSCELoss(configer=configer)

        self.contrast_criterion = PixelContrastLoss(configer=configer)

    def forward(self, preds, target, with_embed=False):
        h, w = target.size(1), target.size(2)

        assert "seg" in preds
        assert "embed" in preds
        seg = preds['seg']
        embedding = preds['embed']

        pred = F.interpolate(input=seg, size=(h, w), mode='bilinear', align_corners=True)
        loss = self.seg_criterion(pred, target)

        _, predict = torch.max(seg, 1)
        loss_contrast = self.contrast_criterion(embedding, target, predict)

        if with_embed is True:
            return loss + self.loss_weight * loss_contrast

        return loss + 0 * loss_contrast  # just a trick to avoid errors in distributed training


class ContrastAuxCELoss(nn.Module, ABC):
    def __init__(self, configer=None):
        super(ContrastAuxCELoss, self).__init__()
        self.configer = configer

        if self.configer.exists('loss', 'params') and 'ce_ignore_index' in self.configer.get('loss', 'params'):
            ignore_index = self.configer.get('loss', 'params')['ce_ignore_index']
        else:
            ignore_index = 255
        Log.info('ignore_index: {}'.format(ignore_index))

        self.loss_weight = self.configer.get('contrast', 'loss_weight')
        self.use_rmi = self.configer.get('contrast', 'use_rmi')

        if self.use_rmi:
            self.seg_criterion = FSAuxRMILoss(configer=configer)
        else:
            self.seg_criterion = FSAuxCELoss(configer=configer)

        self.contrast_criterion = PixelContrastLoss(configer=configer)

    def forward(self, preds, target, with_embed=False):
        h, w = target.size(1), target.size(2)

        assert "seg" in preds
        assert "seg_aux" in preds
        assert "embed" in preds

        seg = preds['seg']
        seg_aux = preds['seg_aux']
        embedding = preds['embed']

        pred = F.interpolate(input=seg, size=(h, w), mode='bilinear', align_corners=True)
        pred_aux = F.interpolate(input=seg_aux, size=(h, w), mode='bilinear', align_corners=True)
        loss = self.seg_criterion([pred_aux, pred], target)

        _, predict = torch.max(seg, 1)
        loss_contrast = self.contrast_criterion(embedding, target, predict)

        if with_embed is True:
            return loss + self.loss_weight * loss_contrast

        return loss + 0 * loss_contrast  # just a trick to avoid errors in distributed training
