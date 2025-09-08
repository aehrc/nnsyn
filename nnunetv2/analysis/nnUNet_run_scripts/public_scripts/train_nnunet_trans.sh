#!/bin/bash

#SBATCH --account=OD-218779
#SBATCH --job-name=training
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --mem=80gb
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --signal=USR1@360
#SBATCH --output=logs/R-%x.%j-data540_folds.out
#SBATCH --open-mode=append


#export NCCL_P2P_DISABLE=1 # solve torch multi-gpu freeze
# export NCCL_P2P_DISABLE=1

# Application specific commands:
module load miniconda3
module load cuda/11.8.0

eval "$(conda shell.bash hook)"
conda activate /datasets/work/hb-synthrad2023/work/synthrad2025/bw_workplace/envs/nnsyn_public

cd /datasets/work/hb-synthrad2023/work/synthrad2025/bw_workplace/ref/nnsyn/nnunetv2/analysis/nnUNet_run_scripts/public_scripts

export nnsyn_origin_dataset="/datasets/work/hb-synthrad2023/work/synthrad2025/bw_workplace/data/nnunet_struct/nnsyn_origin/synthrad2025_task1_mri2ct_AB"
export nnUNet_raw="/datasets/work/hb-synthrad2023/work/synthrad2025/bw_workplace/data/nnunet_struct/raw"
export nnUNet_preprocessed="/datasets/work/hb-synthrad2023/work/synthrad2025/bw_workplace/data/nnunet_struct/preprocessed"
export nnUNet_results="/datasets/work/hb-synthrad2023/work/synthrad2025/bw_workplace/data/nnunet_struct/results"

nnsyn_plan_and_preprocess -d 960 -c 3d_fullres -pl nnUNetPlannerResEncL -p nnUNetResEncUNetLPlans  --preprocessing_input MR --preprocessing_target CT 
nnsyn_plan_and_preprocess_seg -d 960 -dseg 961 -c 3d_fullres -p nnUNetResEncUNetLPlans
git switch nnunetv2
nnUNetv2_train 961 3d_fullres 0 -tr nnUNetTrainer -p nnUNetResEncUNetLPlans_Dataset960 --c
git switch main
nnsyn_train 960 3d_fullres 0 -tr nnUNetTrainer_nnsyn_loss_map -p nnUNetResEncUNetLPlans
# aim up --host 127.0.0.1 --port 43800 --repo /datasets/work/hb-synthrad2023/work/synthrad2025/bw_workplace/data/nnunet_struct/results/runs_aim --workers=4


# unet models
# srun nnUNetv2_train 584 3d_fullres 0 -tr nnUNetTrainerMRCT_track --c
# srun nnUNetv2_train 284 3d_fullres 0 -tr nnUNetTrainerMRCT_1500epochs
# srun nnUNetv2_train 290 3d_fullres 0 -tr nnUNetTrainerMRCT_track -p nnUNetResEncUNetLPlans --c
# srun nnUNetv2_train 544 3d_fullres 0 -tr nnUNetTrainerMRCT_loss_masked
srun nnUNetv2_train 960 3d_fullres 0 -tr nnUNetTrainerMRCT_track -p nnUNetResEncUNetLPlans --c
# srun nnUNetv2_train 264 3d_fullres 0 -tr nnUNetTrainerMRCT_loss_masked_perception -p nnUNetResEncUNetLPlans --c
# srun nnUNetv2_train 264 3d_fullres 0 -tr nnUNetTrainerMRCT_loss_masked_perception_L2_imglossweight0_7 -p nnUNetResEncUNetLPlans --c
# srun nnUNetv2_train 264 3d_fullres 0 -tr nnUNetTrainerMRCT_loss_masked_perception_masked_continue_500epochs -p nnUNetResEncUNetLPlans -pretrained_weights /datasets/work/hb-synthrad2023/work/synthrad2025/bw_workplace/data/nnunet_struct/results/Dataset264_synthrad2025_task1_MR_TH_pre_v2r_stitched_masked/nnUNetTrainerMRCT_loss_masked_perception_masked__nnUNetResEncUNetLPlans__3d_fullres/fold_0/checkpoint_final.pth
srun nnUNetv2_train 960 3d_fullres 0 -tr nnUNetTrainer_nnsyn_loss_masked_perception_masked_track -p nnUNetResEncUNetLPlans --c

# pretained_weights
# fold=1
# srun nnUNetv2_train 240 3d_fullres ${fold} -tr nnUNetTrainerMRCT_loss_masked_perception_masked_continue_500epochs -p nnUNetResEncUNetLPlans -pretrained_weights /datasets/work/hb-synthrad2023/work/synthrad2025/bw_workplace/data/nnunet_struct/results/Dataset260_synthrad2025_task1_MR_AB_pre_v2r_stitched_masked/nnUNetTrainerMRCT_loss_masked_perception_masked__nnUNetResEncUNetLPlans__3d_fullres/fold_${fold}/checkpoint_final.pth




## mednext models
# srun nnUNetv2_train 300 3d_fullres 0 -tr nnUNetTrainerV2_MedNeXt_L_kernel5 --c
# srun nnUNetv2_train 302 3d_fullres_patch_64_192_192 0 -tr nnUNetTrainerV2_MedNeXt_L_kernel5 --c

# segmentation loss
# srun nnUNetv2_train 250 3d_fullres 0 -tr nnUNetTrainerMRCT_loss_masked -p nnUNetResEncUNetLPlans -pretrained_weights /datasets/work/hb-synthrad2023/work/synthrad2025/bw_workplace/data/nnunet_struct/results/Dataset260_synthrad2025_task1_MR_AB_pre_v2r_stitched_masked/nnUNetTrainerMRCT_loss_masked__nnUNetResEncUNetLPlans__3d_fullres/fold_0/checkpoint_final.pth


