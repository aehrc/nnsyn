#!/bin/bash

#SBATCH --account=OD-218779
#SBATCH --job-name=train-seg
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --mem=80gb
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --signal=USR1@360
#SBATCH --output=logs/R-%x.%j-data802.out
#SBATCH --open-mode=append


#export NCCL_P2P_DISABLE=1 # solve torch multi-gpu freeze
# export NCCL_P2P_DISABLE=1

# Application specific commands:
module load miniconda3
module load cuda/11.8.0

eval "$(conda shell.bash hook)"
conda activate /datasets/work/hb-synthrad2023/work/synthrad2025/bw_workplace/envs/nnunet

cd /datasets/work/hb-synthrad2023/work/synthrad2025/bw_workplace/ref/nnUNet_translation/nnunetv2/analysis/nnUNet_run_scripts

export nnUNet_raw="/datasets/work/hb-synthrad2023/work/synthrad2025/bw_workplace/data/nnunet_struct/raw"
export nnUNet_preprocessed="/datasets/work/hb-synthrad2023/work/synthrad2025/bw_workplace/data/nnunet_struct/preprocessed"
export nnUNet_results="/datasets/work/hb-synthrad2023/work/synthrad2025/bw_workplace/data/nnunet_struct/results"


## segmentation models
srun nnUNetv2_train 801 3d_fullres 0 -tr nnUNetTrainer -p nnUNetResEncUNetLPlans_Dataset263 --c