#!/bin/bash

#SBATCH -p gpu --gres=gpu:1
#SBATCH -n 1
#SBATCH --mem=16G
#SBATCH -t 24:00:00

module load cuda

source /gpfs/runtime/opt/anaconda/2020.02/etc/profile.d/conda.sh
conda activate pt
# cd /gpfs/data/ssrinath/sding13/deep-visual-hull/
# python run.py --load_vgg --num_epoches 200 --save_dir /gpfs/sding13/scratch/checkpoints/


python /users/sding13/data/sding13/deep-visual-hull/run.py --num_epoches 30 --save_dir /users/sding13/scratch/checkpoints