import random
from src.model import *
from lib.utils.tools.configer import Configer
from src.trainer import Trainer
from src.model.nets import get_model
from src.optimizer import get_optimizer, get_scheduler
from lib.dataset.dataset import get_val_loader, get_train_loader
import argparse
from src.util import load_cfg_from_cfg_file, merge_cfg_from_list
from lib.utils.tools.logger import Logger as Log

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
from mask import Masker
masker = Masker()

# =================== get config ===================
"""  
DATA= pascal 
SPLIT= 0
GPU = [0]
LAYERS= 50 
SHOT= 1                         
"""

arg_input = ' --config config_files/pascal_pretrain.json '


def str2bool(v):
    """ Usage:
    parser.add_argument('--pretrained', type=str2bool, nargs='?', const=True,
                        dest='pretrained', help='Whether to use pretrained models.')
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
    parser.add_argument('--gpu', default=None, nargs='+', type=int,
                        dest='gpu', help='The gpu list used.')

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
    parser.add_argument('--group_method', default=None, type=str,
                        dest='optim:group_method', help='The group method that used.')
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

    args_parser = parser.parse_args(arg_input.split())
    return args_parser

args = parse_args()

configer = Configer(args_parser=args)

print(args.__dict__)

# ====================================  main ================================================
random.seed(args.manual_seed)
np.random.seed(args.manual_seed)
torch.manual_seed(args.manual_seed)

# ====== Model + Optimizer ======
trainer = Trainer(configer = configer)
modules_ori = [model.layer0, model.layer1, model.layer2, model.layer3, model.layer4]
modules_new = [model.ppm, model.bottleneck, model.classifier]
if args.contrast:
    modules_new.append(model.proj_head)


params_list = []
for module in modules_ori:
    params_list.append(dict(params=module.parameters(), lr=args.lr))
for module in modules_new:
    params_list.append(dict(params=module.parameters(), lr=args.lr * args.scale_lr))
optimizer = get_optimizer(args, params_list)

validate_fn = episodic_validate if args.episodic_val else standard_validate

# ========= Data  ==========
args.workers = 0
args.batch_size=2
args.distributed = False
train_loader, train_sampler = get_train_loader(args, episodic=False)
val_loader, _ = get_val_loader(args,  episodic=args.episodic_val)  # mode='train' means that we will validate on images from validation set, but with the bases classes

# ========== Scheduler  ================
scheduler = get_scheduler(args, optimizer, len(train_loader))

# ====== Metrics initialization ======
max_val_mIoU = 0.
iter_per_epoch = len(train_loader)
log_iter = int(iter_per_epoch / args.log_freq) + 1

# ================================================ Training ================================================
iterable_train_loader = iter(train_loader,)
model.train()

images, gt = iterable_train_loader.next()  # q: [1, 3, 473, 473], s: [1, 1, 3, 473, 473]
criterion = ContrastCELoss(args=args)
logits, embedding = model(images)


loss = criterion(logits=logits, target=gt.long(), embedding=embedding, with_embed=True)

sv_path = 'pretrain_{}'.format(args.train_name) + ('_contrast' if args.contrast else '_no') + \
          '/{}{}/split{}_shot{}/{}'.format(args.arch, args.layers, args.train_split, args.shot, args.exp_name)