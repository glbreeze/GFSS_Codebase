##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Created by: RainbowSecret, JingyiXie, LangHuang
## Microsoft Research
## yuyua@microsoft.com
## Copyright (c) 2019
##
## This source code is licensed under the MIT-style license found in the
## LICENSE file in the root directory of this source tree
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch.nn.parallel
import torch.utils.data
import torch.nn.functional as F

from lib.dataset.dataset import get_val_loader, get_train_loader
from .util import intersectionAndUnionGPU, AverageMeter, get_model_dir_trans

import time
import torch

from lib.utils.tools.average_meter import AverageMeter
from lib.loss.loss_manager import LossManager
from lib.models.model_manager import ModelManager
from lib.utils.tools.logger import Logger as Log
from .tools.module_runner import ModuleRunner
from .tools.optim_scheduler import OptimScheduler
from lib.utils.distributed import get_world_size, get_rank, is_distributed

class Trainer(object):

    def __init__(self, configer):
        self.configer = configer
        self.batch_time = AverageMeter()
        self.train_losses = AverageMeter()
        self.val_losses = AverageMeter()

        self.loss_manager = LossManager(configer)
        self.module_runner = ModuleRunner(configer)
        self.model_manager = ModelManager(configer=configer)

        self.optim_scheduler = OptimScheduler(configer)

        self.seg_net = None
        self.train_loader = None
        self.val_loader = None
        self.optimizer = None
        self.scheduler = None
        self.running_score = None

        self._init_model()

    def _init_model(self):
        self.seg_net = self.model_manager.semantic_segmentor()   # ======================================== define model

        try:
            flops, params = get_model_complexity_info(self.seg_net, (3, 512, 512))
            split_line = '=' * 30
            print('{0}\nInput shape: {1}\nFlops: {2}\nParams: {3}\n{0}'.format(
                split_line, (3, 512, 512), flops, params))
            print('!!!Please be cautious if you use the results in papers. '
                  'You may need to check if all ops are supported and verify that the '
                  'flops computation is correct.')
        except:
            pass

        self.seg_net = self.module_runner.load_net(self.seg_net)

        self.train_loader, _ = get_train_loader(self.configer, episodic=False)
        self.val_loader, _ = get_val_loader(self.configer, episodic=self.configer.get('val', 'episodic_val'))
        self.configer.add(['solver', 'max_iters'], self.configer.get('solver','max_epoch')*len(self.train_loader))
        Log.info('total iterations {}'.format(self.configer.get('solver', 'max_iters')))

        params_group = self._get_parameters()
        self.optimizer, self.scheduler = self.optim_scheduler.init_optimizer(params_group)

        self.pixel_loss = self.loss_manager.get_seg_loss()
        if is_distributed():
            self.pixel_loss = self.module_runner.to_device(self.pixel_loss)

    def _get_parameters(self):
        bb_lr = []
        nbb_lr = []
        fcn_lr = []
        params_dict = dict(self.seg_net.named_parameters())
        for key, value in params_dict.items():
            if 'backbone' in key:
                bb_lr.append(value)
            elif 'aux_layer' in key or 'upsample_proj' in key:
                fcn_lr.append(value)
            else:
                nbb_lr.append(value)

        params = [{'params': bb_lr, 'lr': self.configer.get('lr', 'base_lr')},
                  {'params': fcn_lr, 'lr': self.configer.get('lr', 'base_lr') * 10},
                  {'params': nbb_lr, 'lr': self.configer.get('lr', 'base_lr') * self.configer.get('lr', 'nbb_mult')}]
        return params

    def __train(self):
        """
          Train function of every epoch during train phase.
        """
        self.seg_net.train()
        self.pixel_loss.train()
        start_time = time.time()
        scaler = torch.cuda.amp.GradScaler()

        if hasattr(self.train_loader.sampler, 'set_epoch'):  # for distributed training
            self.train_loader.sampler.set_epoch(self.configer.get('epoch'))

        for i, data_dict in enumerate(self.train_loader):
            images, gt = data_dict
            if torch.cuda.is_available():
                images = images.cuda()  # [1, 1, 3, h, w]
                gt = gt.cuda()  # [1, 1, h, w]

            with torch.cuda.amp.autocast():
                outputs = self.seg_net(images)

            if is_distributed():
                import torch.distributed as dist
                def reduce_tensor(inp):
                    """
                    Reduce the loss from all processes so that
                    process with rank 0 has the averaged results.
                    """
                    world_size = get_world_size()
                    if world_size < 2:
                        return inp
                    with torch.no_grad():
                        reduced_inp = inp
                        dist.reduce(reduced_inp, dst=0)
                    return reduced_inp

                with torch.cuda.amp.autocast():
                    loss = self.pixel_loss(outputs, gt)
                    backward_loss = loss
                    display_loss = reduce_tensor(backward_loss) / get_world_size()
            else:
                backward_loss = display_loss = self.pixel_loss(outputs, gt)

            scaler.scale(backward_loss).backward()
            scaler.step(self.optimizer)
            scaler.update()
            self.optimizer.zero_grad(set_to_none=True)

            if self.configer.get('lr', 'metric') == 'iters':
                self.scheduler.step()
            if self.configer.get('lr', 'is_warm'):
                self.module_runner.warm_lr(
                    self.configer.get('iters'), self.scheduler, self.optimizer, backbone_list=[0, ]
                )

            # Update the vars of the train phase.
            self.train_losses.update(display_loss.item(), self.configer.get("train", "batch_size"))  # running avg of loss
            self.configer.plus_one('iters')
            self.batch_time.update(time.time() - start_time)
            start_time = time.time()

            # Print the log info & reset the states.
            if self.configer.get('iters') % self.configer.get('solver', 'display_iter') == 0 and (not is_distributed() or get_rank() == 0):
                Log.info('Train Epoch: {0}, Train Iteration: {1}, '
                         'Time {batch_time.sum:.3f}s/{2}iters, ({batch_time.avg:.3f}) '
                         'Learning rate = {3}, Loss = {loss.val:.4f} (ave = {loss.avg:.4f})\n'.format(
                    self.configer.get('epoch'), self.configer.get('iters'), self.configer.get('solver', 'display_iter'),
                    [round(e, 6) for e in self.module_runner.get_lr(self.optimizer)], batch_time=self.batch_time, loss=self.train_losses))
                self.batch_time.reset()
                self.train_losses.reset()

            # Check to val the current model.
            if self.configer.get('iters') % self.configer.get('solver', 'test_interval') == 0:
                val_Iou = self.standard_validate()

        self.configer.plus_one('epoch')
        if self.configer.get('lr', 'metric') != 'iters':
            self.scheduler.step()

    def train(self):
        # cudnn.benchmark = True
        # self.__val()
        if self.configer.get('network', 'resume') is not None:
            if self.configer.get('network', 'resume_val'):
                self.__val(data_loader=self.data_loader.get_valloader(dataset='val'))
                return
            elif self.configer.get('network', 'resume_train'):
                self.__val(data_loader=self.data_loader.get_valloader(dataset='train'))
                return

        while self.configer.get('epoch') < self.configer.get('solver', 'max_epoch'):
            self.__train()

        # use swa to average the model
        if 'swa' in self.configer.get('lr', 'lr_policy'):
            self.optimizer.swap_swa_sgd()
            self.optimizer.bn_update(self.train_loader, self.seg_net)

        self.standard_validate()

    def standard_validate(self, val_loader=None):
        self.seg_net.eval()
        self.pixel_loss.eval()
        start_time = time.time()

        num_classes_tr = self.configer.get('data', 'num_classes')
        intersections, unions = torch.zeros(num_classes_tr).cuda(), torch.zeros(num_classes_tr).cuda()

        val_loader = self.val_loader if val_loader is None else val_loader
        for i, (images, gt) in enumerate(val_loader):
            if torch.cuda.is_available():
                images = images.cuda()  # [200, 3, 473, 473]
                gt = gt.cuda()          # [200, 473, 473]

            with torch.no_grad():
                logits = self.seg_net(images)['seg'].detach()
                logits = F.interpolate(logits, size=gt.shape[-2:], mode='bilinear', align_corners=True)
                loss_fn = torch.nn.CrossEntropyLoss(ignore_index=255)
                loss = loss_fn(logits, gt)
                self.val_losses.update(loss.item(), self.configer.get('val', 'batch_size'))

            intersection, union, target = intersectionAndUnionGPU(logits.argmax(1), gt, num_classes_tr, 255)  # gt [B, 473, 473]
            intersections += intersection  # [16]
            unions += union                # [16]

            self.batch_time.update(time.time() - start_time)
            start_time = time.time()

        mIoU = (intersections / (unions + 1e-10)).mean()
        acc = intersections.sum() / unions.sum()
        self.configer.update(['val_loss'], self.val_losses.avg)
        self.configer.update(['performance'], mIoU)
        self.module_runner.save_net(self.seg_net, save_mode='val_loss')
        self.module_runner.save_net(self.seg_net, save_mode='performance')
        Log.info(f'====>Test Time {self.batch_time.avg:.2f}s/{self.batch_time.sum:.2f}s: '
                 f'running loss {self.val_losses.avg:.2f}, Acc {acc:.4f}, mIoU {mIoU:.4f}\n')

        self.seg_net.train()
        self.pixel_loss.train()

        self.batch_time.reset()
        self.val_losses.reset()

        return mIoU


if __name__ == "__main__":
    pass
