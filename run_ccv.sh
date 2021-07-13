#!/bin/bash

#SBATCH -p gpu --gres=gpu:1
#SBATCH -n 1
#SBATCH --mem=16G
#SBATCH -t 24:00:00

module load cuda

source /gpfs/runtime/opt/anaconda/2020.02/etc/profile.d/conda.sh
conda activate deep-visual-hull
# cd /gpfs/data/ssrinath/qwei3/deep-visual-hull/
python run.py --num_epochs 500 --save_dir checkpoints --mode train

