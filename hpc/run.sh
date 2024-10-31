#!/bin/bash
#SBATCH --job-name=cae
#SBATCH --partition=dgx_normal_q
#SBATCH --account=niche_squad
#SBATCH --time=120:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --array=0-2
#SBATCH --output=logs/%x-%A-%a.out
#SBATCH --error=logs/%x-%A-%a.err

source activate tf
source .env

python3.9 train.py --job $SLURM_ARRAY_TASK_ID