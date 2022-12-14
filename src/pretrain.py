# encoding:utf-8

# This code is to pretrain model backbone on the base train data
#import pdb
import os
import yaml
import time
import random
import argparse
import torch.backends.cudnn as cudnn
import torch.utils.data
from .trainer import Trainer
# from tensorboardX import SummaryWriter

from .util import ensure_path, set_log_path, log
from lib.utils.tools.configer import Configer
from lib.utils.tools.logger import Logger as Log

def str2bool(v):
    """ Usage:
    parser.add_argument('--pretrained', type=str2bool, nargs='?', const=True,  dest='pretrained', help='Whether to use pretrained models.')
    """
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported value encountered.')


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--configs', default=None, type=str,
                        dest='configs', help='The file of the hyper parameters.')
    parser.add_argument('--phase', default='train', type=str,
                        dest='phase', help='The phase of module.')
    parser.add_argument('--gpu', default=[0], nargs='+', type=int,
                        dest='gpu', help='The gpu list used.')
    parser.add_argument('--train_split', default=0, type=int,
                        dest='train_split', help='data split.')

    # ***********  Params for model.  **********
    parser.add_argument('--model_name', default=None, type=str,
                        dest='network:model_name', help='The name of model.')
    parser.add_argument('--backbone', default=None, type=str,
                        dest='network:backbone', help='The base network of model.')
    parser.add_argument('--bn_type', default=None, type=str,
                        dest='network:bn_type', help='The BN type of the network.')
    parser.add_argument('--multi_grid', default=None, nargs='+', type=int,
                        dest='network:multi_grid', help='The multi_grid for resnet backbone.')
    parser.add_argument('--pretrained', type=str, default=None,
                        dest='network:pretrained', help='The path to pretrained model.')
    parser.add_argument('--resume', default=None, type=str,
                        dest='network:resume', help='The path of checkpoints.')
    parser.add_argument('--resume_strict', type=str2bool, nargs='?', default=True,
                        dest='network:resume_strict', help='Fully match keys or not.')
    parser.add_argument('--resume_continue', type=str2bool, nargs='?', default=False,
                        dest='network:resume_continue', help='Whether to continue training.')
    parser.add_argument('--resume_eval_train', type=str2bool, nargs='?', default=True,
                        dest='network:resume_train', help='Whether to validate the training set  during resume.')
    parser.add_argument('--resume_eval_val', type=str2bool, nargs='?', default=True,
                        dest='network:resume_val', help='Whether to validate the val set during resume.')
    parser.add_argument('--gathered', type=str2bool, nargs='?', default=True,
                        dest='network:gathered', help='Whether to gather the output of model.')
    parser.add_argument('--loss_balance', type=str2bool, nargs='?', default=False,
                        dest='network:loss_balance', help='Whether to balance GPU usage.')

    # ***********  Params for solver.  **********
    parser.add_argument('--optim_method', default=None, type=str,
                        dest='optim:optim_method', help='The optim method that used.')
    parser.add_argument('--base_lr', default=None, type=float,
                        dest='lr:base_lr', help='The learning rate.')
    parser.add_argument('--nbb_mult', default=1.0, type=float,
                        dest='lr:nbb_mult', help='The not backbone mult ratio of learning rate.')
    parser.add_argument('--lr_policy', default=None, type=str,
                        dest='lr:lr_policy', help='The policy of lr during training.')
    parser.add_argument('--loss_type', default=None, type=str,
                        dest='loss:loss_type', help='The loss type of the network.')
    parser.add_argument('--is_warm', type=str2bool, nargs='?', default=False,
                        dest='lr:is_warm', help='Whether to warm training.')

    # ***********  Params for display.  **********
    parser.add_argument('--max_epoch', default=None, type=int,
                        dest='solver:max_epoch', help='The max epoch of training.')
    parser.add_argument('--max_iters', default=None, type=int,
                        dest='solver:max_iters', help='The max iters of training.')
    parser.add_argument('--display_iter', default=None, type=int,
                        dest='solver:display_iter', help='The display iteration of train logs.')
    parser.add_argument('--test_interval', default=None, type=int,
                        dest='solver:test_interval', help='The test interval of validation.')

    # ***********  Params for env.  **********
    parser.add_argument('--seed', default=304, type=int, help='manual seed')
    parser.add_argument('--cudnn', type=str2bool, nargs='?', default=True, help='Use CUDNN.')

    # ***********  Params for distributed training.  **********
    parser.add_argument('--local_rank', type=int, default=-1, dest='local_rank', help='local rank of current process')
    parser.add_argument('--distributed', action='store_true', dest='distributed', help='Use multi-processing training.')
    parser.add_argument('--use_ground_truth', action='store_true', dest='use_ground_truth',
                        help='Use ground truth for training.')

    parser.add_argument('REMAIN', nargs='*')

    args_parser = parser.parse_args()
    return args_parser


if __name__ == "__main__":
    args_parser = parse_args()

    from lib.utils.distributed import handle_distributed
    handle_distributed(args_parser, os.path.expanduser(os.path.abspath(__file__)))

    if args_parser.seed is not None:
        random.seed(args_parser.seed)
        torch.manual_seed(args_parser.seed)

    cudnn.enabled = True
    cudnn.benchmark = args_parser.cudnn

    configer = Configer(args_parser=args_parser)
    project_dir = os.path.dirname(os.path.realpath(__file__))
    configer.add(['project_dir'], project_dir)

    sv_path = './results/pretrain_{0}/split{1}/{2}/{3}'.format(
        configer.get('train_name'), configer.get('train_split'), configer.get('network','model_name'), configer.get('exp_name'))
    ensure_path(sv_path)
    configer.add(['sv_path'], sv_path)

    log_fname = '{}_{}.txt'.format('log', time.strftime("%Y-%m-%d_%X", time.localtime()))
    Log.init(logfile_level='info', stdout_level='info',
             log_file=os.path.join(sv_path, log_fname),
             rewrite=True)

    model = Trainer(configer)             # ============================================================== define model
    model.train()
