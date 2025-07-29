#!/bin/bash

#SBATCH --account=OD-218779
#SBATCH --job-name=preprocessing
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --mem=40gb
#SBATCH --cpus-per-task=12
#SBATCH --output=logs/R-%x.%j-preprocessing-AB.out


#export NCCL_P2P_DISABLE=1 # solve torch multi-gpu freeze
# export NCCL_P2P_DISABLE=1

# Application specific commands:
module load miniconda3
module load cuda/11.8.0

eval "$(conda shell.bash hook)"
conda activate /datasets/work/hb-synthrad2023/work/synthrad2025/bw_workplace/envs/nnunet_trans2

cd /datasets/work/hb-synthrad2023/work/synthrad2025/bw_workplace/ref/preprocessing
srun python stage2_task1.py
