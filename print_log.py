
import os
import argparse
import numpy as np
from collections import defaultdict
from src.util import AverageMeter


parser = argparse.ArgumentParser(description='Summarize log')
parser.add_argument('--fpath', type=str, required=True, help='path of log file')
args = parser.parse_args()

for file in os.listdir(args.fpath):
    if file.endswith('.txt'):
        fname = file
fpath = os.path.join(args.fpath, fname)
assert os.path.exists(fpath), 'can not find the file'

eval_cnt = 0
iou_lst = []
avg_iou_lst = []

f = open(fpath, 'r')
line = f.readline()
while line:
    if "====>Test" in line:
        eval_cnt += 1

        iou = line.split()[-1]
        iou_lst.append(float(iou))

        new_line = line[30:].strip() + ', avg iou {:.4f}, max iou {:.4f},'.format(np.mean(iou_lst), np.max(iou_lst))

        if eval_cnt>=4:
            smt_iou = np.mean(iou_lst[-4:])
            avg_iou_lst.append(smt_iou)
            new_line = new_line + 'smt iou {:.4f}, max smt iou {:.4f}'.format(smt_iou, np.max(avg_iou_lst))

        print(new_line)

    line = f.readline()

f.close()
