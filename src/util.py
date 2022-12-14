# encoding:utf-8

import os
import shutil
import torch
import torch.nn.functional as F
import yaml
import copy
from ast import literal_eval
from typing import Callable, Iterable, List, TypeVar
import torch.distributed as dist
from typing import Tuple
import argparse
import numpy as np

A = TypeVar("A")
B = TypeVar("B")


def tensor_slice(x, idx=None, ref=None):    # idx is 0/1, ref is tensor on gpu
    n, c, h, w = x.shape
    x = x.view(n // 2, 2, c, h, w)
    if idx is not None:
        out = x[:, idx, :, :, :]
    elif ref is not None:
        ref = ref.reshape((len(ref)//2, 2))
        indices = torch.argmax(ref, dim=-1)
        out = x[ torch.arange(n//2, device=indices.device), indices ]
    return out


def get_aux_loss(wt, att_q, f_q, q_label, model, eps=0.6, reduction='mean'):
    pd0 = F.softmax(model.classifier(att_q), dim=1)   # [1, 2, 60, 60]
    pd1 = F.softmax(model.classifier(f_q), dim=1)

    label = F.interpolate(q_label.unsqueeze(1).float(), size=pd0.shape[-2:], mode='nearest').squeeze(1)
    label[label > 1] = 255

    det0 = torch.abs(pd0[:, 1, :, :] - label).data
    det1 = torch.abs(pd1[:, 1, :, :] - label).data

    loss_lhs = (wt[:, 0, :, :] - wt[:, 1, :, :]) * torch.sign(det0 - det1)
    loss_rhs = -eps * torch.abs(det0 - det1)
    loss_aux = torch.maximum(loss_lhs, loss_rhs)

    loss_aux = torch.mean(loss_aux)
    return loss_aux


def get_wt_loss(wt, att_q, f_q, q_label, model, eps=0.03, reduction='mean'):
    pd0 = model.classifier(att_q)  # [1, 2, 60, 60]
    pd1 = model.classifier(f_q)
    label = F.interpolate(q_label.unsqueeze(1).float(), size=pd0.shape[-2:], mode='nearest')
    label[label > 1] = 255

    ce_loss = torch.nn.CrossEntropyLoss(ignore_index=255, reduction='none')
    loss0 = ce_loss(pd0, label.squeeze(1).long()).data
    loss1 = ce_loss(pd1, label.squeeze(1).long()).data

    delta = loss0 - loss1  # [1, 60, 60]
    mask = (delta < 0).long()  # att 优于 f_q
    mask[mask == 0] = -1  # [1, 60, 60]
    wt10 = wt[0, 1:2, :, :] - wt[0, 0:1, :, :] - eps  # [1, 60, 60]   f_q weight - att weight

    wt10 = wt10 * mask
    wt_loss = torch.maximum(wt10, torch.tensor(0.0).cuda())
    if reduction == 'mean':
        return torch.mean(wt_loss)
    elif reduction == 'none':
        return wt_loss


def ensure_path(path, remove=True):
    if os.path.exists(path):
        if remove or input('{} exists, remove? ([y]/n): '.format(path)) != 'n':
            print('remove the existing folder {}'.format(path))
            shutil.rmtree(path)

    os.makedirs(path)


def set_log_path(path):
    global _log_path
    _log_path = path


def log(obj, filename='log.txt'):
    print(obj)
    if _log_path is not None:
        with open(os.path.join(_log_path, filename), 'a') as f:
            print(obj, file=f)


def get_mid_feat(model, x, layer='4'):   # x: input to model
    feat_blobs = []

    def hook_feature(module, input, output):
        feat_blobs.append(output)

    handles = {}
    handles[4] = model.layer4[2].bn3.register_forward_hook(hook_feature)

    with torch.no_grad():
        f, f_lst = model.extract_features(x)
        feat = feat_blobs[-1]                  # 需要输出的feature
        for k, v in handles.items():
            handles[k].remove()

    return f, f_lst, [feat]


def setup(
        args: argparse.Namespace,
        rank: int,
        world_size: int
) -> None:
    """
    Used for distributed learning
    """
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = str(args.port)

    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def cleanup() -> None:
    """
    Used for distributed learning
    """
    dist.destroy_process_group()


def find_free_port() -> int:
    """
    Used for distributed learning
    """
    import socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("", 0))
    port = sock.getsockname()[1]
    sock.close()
    return port


def map_(fn: Callable[[A], B], iter: Iterable[A]) -> List[B]:
    """
    Used for multiprocessing
    """
    return list(map(fn, iter))


def get_model_dir(args: argparse.Namespace) -> str:
    """
    Obtain the directory to save/load the model
    """
    path = os.path.join(
        args.model_dir,
        args.train_name,
        f'split={args.train_split}',
        'model',
        f'shot_{args.shot}',
        f'pspnet_{args.arch}{args.layers}'
    )
    return path


def get_model_dir_trans(args: argparse.Namespace) -> str:  # model_ckpt/coco/split=0/model/shot_1/transformer_resnet50
    """
    Obtain the directory to save/load the model
    """
    path = os.path.join(
        args.model_dir,
        args.train_name,
        f'split={args.train_split}',
        'model',
        f'shot_{args.shot}',
        f'transformer_{args.arch}{args.layers}'
    )
    return path


def to_one_hot(mask: torch.tensor, num_classes: int) -> torch.tensor:
    """
    inputs:
        mask : shape [n_task, shot, h, w]
        num_classes : Number of classes

    returns :
        one_hot_mask : shape [n_task, shot, num_class, h, w]
    """
    n_tasks, shot, h, w = mask.size()
    one_hot_mask = torch.zeros(n_tasks, shot, num_classes, h, w).to(dist.get_rank())
    new_mask = mask.unsqueeze(2).clone()
    new_mask[torch.where(new_mask == 255)] = 0  # Ignore_pixels are anyways filtered out in the losses
    one_hot_mask.scatter_(2, new_mask, 1).long()
    return one_hot_mask


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class CompareMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.cnt = 0
        self.win_cnt = 0
        self.diff_sum = 0
        self.diff_avg = 0

    def update(self, score1, score0):
        self.val = (score1, score0)
        self.cnt += 1
        self.win_cnt += (score1 >score0 )
        self.diff_sum += score1-score0
        self.diff_avg = self.diff_sum / self.cnt


def batch_intersectionAndUnionGPU(
        logits: torch.Tensor,
        target: torch.Tensor,
        num_classes: int,
        ignore_index=255
) -> Tuple[torch.tensor, torch.tensor, torch.tensor]:
    """
    inputs:
        logits : shape [n_task, shot, num_class, h, w]
        target : shape [n_task, shot, H, W]
        num_classes : Number of classes

    returns :
        area_intersection : shape [n_task, shot, num_class]
        area_union : shape [n_task, shot, num_class]
        area_target : shape [n_task, shot, num_class]
    """
    n_task, shots, num_classes, h, w = logits.size()
    H, W = target.size()[-2:]

    logits = F.interpolate(
        logits.view(n_task * shots, num_classes, h, w),
        size=(H, W), mode='bilinear', align_corners=True
    ).view(n_task, shots, num_classes, H, W)

    preds = logits.argmax(2)  # [n_task, shot, H, W]

    n_tasks, shot, num_classes, H, W = logits.size()
    area_intersection = torch.zeros(n_tasks, shot, num_classes)
    area_union = torch.zeros(n_tasks, shot, num_classes)
    area_target = torch.zeros(n_tasks, shot, num_classes)
    for task in range(n_tasks):
        for shot in range(shots):
            i, u, t = intersectionAndUnionGPU(
                preds[task][shot], target[task][shot],
                num_classes, ignore_index=ignore_index
            ) # i,u, t are of size()
            area_intersection[task, shot, :] = i
            area_union[task, shot, :] = u
            area_target[task, shot, :] = t
    return area_intersection, area_union, area_target


def intersectionAndUnionGPU(
        preds: torch.tensor,
        target: torch.tensor,
        num_classes: int,
        ignore_index=255
) -> Tuple[torch.tensor, torch.tensor, torch.tensor]:
    """
    inputs:
        preds : shape [H, W]
        target : shape [H, W]
        num_classes : Number of classes

    returns :
        area_intersection : shape [num_class]
        area_union : shape [num_class]
        area_target : shape [num_class]
    """
    assert (preds.dim() in [1, 2, 3])
    assert preds.shape == target.shape
    preds = preds.view(-1)
    target = target.view(-1)
    preds[target == ignore_index] = ignore_index
    intersection = preds[preds == target]
    area_intersection = torch.histc(intersection.float(), bins=num_classes, min=0, max=num_classes-1)
    area_output = torch.histc(preds.float(), bins=num_classes, min=0, max=num_classes-1)
    area_target = torch.histc(target.float(), bins=num_classes, min=0, max=num_classes-1)
    area_union = area_output + area_target - area_intersection
    # print(torch.unique(intersection))
    return area_intersection, area_union, area_target


# ======================================================================================================================
# ======== All following helper functions have been borrowed from from https://github.com/Jia-Research-Lab/PFENet ======
# ======================================================================================================================

class CfgNode(dict):
    """
    CfgNode represents an internal node in the configuration tree. It's a simple
    dict-like container that allows for attribute-based access to keys.
    """

    def __init__(self, init_dict=None, key_list=None, new_allowed=False):
        # Recursively convert nested dictionaries in init_dict into CfgNodes
        init_dict = {} if init_dict is None else init_dict
        key_list = [] if key_list is None else key_list
        for k, v in init_dict.items():
            if type(v) is dict:
                # Convert dict to CfgNode
                init_dict[k] = CfgNode(v, key_list=key_list + [k])
        super(CfgNode, self).__init__(init_dict)

    def __getattr__(self, name):
        if name in self:
            return self[name]
        else:
            raise AttributeError(name)

    def __setattr__(self, name, value):
        self[name] = value

    def __str__(self):
        def _indent(s_, num_spaces):
            s = s_.split("\n")
            if len(s) == 1:
                return s_
            first = s.pop(0)
            s = [(num_spaces * " ") + line for line in s]
            s = "\n".join(s)
            s = first + "\n" + s
            return s

        r = ""
        s = []
        for k, v in sorted(self.items()):
            seperator = "\n" if isinstance(v, CfgNode) else " "
            attr_str = "{}:{}{}".format(str(k), seperator, str(v))
            attr_str = _indent(attr_str, 2)
            s.append(attr_str)
        r += "\n".join(s)
        return r

    def __repr__(self):
        return "{}({})".format(self.__class__.__name__, super(CfgNode, self).__repr__())


def _decode_cfg_value(v):
    if not isinstance(v, str):
        return v
    try:
        v = literal_eval(v)
    except ValueError:
        pass
    except SyntaxError:
        pass
    return v


def _check_and_coerce_cfg_value_type(replacement, original, key, full_key):
    original_type = type(original)
    replacement_type = type(replacement)

    # The types must match (with some exceptions)
    if replacement_type == original_type:
        return replacement

    def conditional_cast(from_type, to_type):
        if replacement_type == from_type and original_type == to_type:
            return True, to_type(replacement)
        else:
            return False, None

    casts = [(tuple, list), (list, tuple)]
    try:
        casts.append((str, unicode))  # noqa: F821
    except Exception:
        pass

    for (from_type, to_type) in casts:
        converted, converted_value = conditional_cast(from_type, to_type)
        if converted:
            return converted_value

    raise ValueError(
        "Type mismatch ({} vs. {}) with values ({} vs. {}) for config "
        "key: {}".format(
            original_type, replacement_type, original, replacement, full_key
        )
    )


def load_cfg_from_cfg_file(file: str):
    cfg = {}
    assert os.path.isfile(file) and file.endswith('.yaml'), \
        '{} is not a yaml file'.format(file)

    with open(file, 'r') as f:
        cfg_from_file = yaml.safe_load(f)

    for key in cfg_from_file:
        for k, v in cfg_from_file[key].items():
            cfg[k] = v

    cfg = CfgNode(cfg)
    return cfg


def merge_cfg_from_list(cfg: CfgNode, cfg_list: List[str]):
    new_cfg = copy.deepcopy(cfg)
    assert len(cfg_list) % 2 == 0, cfg_list
    for full_key, v in zip(cfg_list[0::2], cfg_list[1::2]):
        subkey = full_key.split('.')[-1]
        assert subkey in cfg, 'Non-existent key: {}'.format(full_key)
        value = _decode_cfg_value(v)
        value = _check_and_coerce_cfg_value_type(
            value, cfg[subkey], subkey, full_key
        )
        setattr(new_cfg, subkey, value)

    return new_cfg
