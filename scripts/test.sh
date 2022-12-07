#!/bin/bash

#SBATCH --job-name=seg
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=20GB
#SBATCH --time=24:00:00
#SBATCH --mail-type=END
#SBATCH --mail-user=lg154@nyu.edu
#SBATCH --output=seg.out
#SBATCH --gres=gpu # How much gpu need, n is the number


module purge

DATA=$1
SPLIT=$2
LAYERS=$3
SHOT=$4

dirname="results/test/resnet-${LAYERS}/${DATA}/split_${SPLIT}"
mkdir -p -- "$dirname"



echo "start"
singularity exec --nv \
            --overlay /scratch/lg154/python36/python36.ext3:ro \
            --overlay /scratch/lg154/sseg/dataset/coco2014.sqf:ro \
            /scratch/work/public/singularity/cuda11.2.2-cudnn8-devel-ubuntu20.04.sif \
            /bin/bash -c " source /ext3/env.sh;
            python -m src.test_ida --config config_files/${DATA}.yaml \
					 --opts train_split ${SPLIT} \
					      batch_size_val 1 \
						    layers ${LAYERS} \
						    shot ${SHOT} \
						    heads 4 \
						    cls_lr 0.1 \
						    test_num 1000 \
					 > ${dirname}/log_${SHOT}.txt 2>&1"

echo "finish"