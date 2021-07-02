#!/bin/bash

#SBATCH -p gpu --gres=gpu:1
#SBATCH -n 1
#SBATCH --mem=16G
#SBATCH -t 24:00:00

module load cuda

source /gpfs/runtime/opt/anaconda/2020.02/etc/profile.d/conda.sh
conda activate deep-visual-hull

python run.py --load_ckpt_dir /gpfs/data/ssrinath/anatara7/deep-visual-hull/checkpoints/1624836375/ --num_epoches 200 --save_dir /gpfs/data/ssrinath/anatara7/deep-visual-hull/checkpoints/


