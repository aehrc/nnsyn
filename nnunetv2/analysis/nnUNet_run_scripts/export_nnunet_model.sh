module load miniconda3
module load cuda/11.8.0

eval "$(conda shell.bash hook)"
conda activate /datasets/work/hb-synthrad2023/work/synthrad2025/bw_workplace/envs/nnunet_trans2

cd /datasets/work/hb-synthrad2023/work/synthrad2025/bw_workplace/ref/nnUNet_translation/nnunetv2/analysis/nnUNet_run_scripts

export nnUNet_raw="/datasets/work/hb-synthrad2023/work/synthrad2025/bw_workplace/data/nnunet_struct/raw"
export nnUNet_preprocessed="/datasets/work/hb-synthrad2023/work/synthrad2025/bw_workplace/data/nnunet_struct/preprocessed"
export nnUNet_results="/datasets/work/hb-synthrad2023/work/synthrad2025/bw_workplace/data/nnunet_struct/results"


# touch /datasets/work/hb-synthrad2023/work/synthrad2025/bw_workplace/data/nnunet_struct/results/Dataset282_synthrad2025_task1_MR_HN_pre_v2r_stitched/nnUNetTrainerMRCT_track__nnUNetPlans__3d_fullres/fold_0/validation/summary.json
DATASET_ID=540
FOLD="0"
save_path="/datasets/work/hb-synthrad2023/work/synthrad2025/bw_workplace/data/nnunet_struct/export_models/testing_dataset${DATASET_ID}_fold${FOLD}"
mkdir -p ${save_path}
nnUNetv2_export_model_to_zip -d $DATASET_ID -o ${save_path}/dataset${DATASET_ID}_fold${FOLD}.zip -c 3d_fullres -tr nnUNetTrainerMRCT_loss_masked_perception_masked -f ${FOLD} -p nnUNetResEncUNetLPlans
# nnUNetv2_export_model_to_zip -d $DATASET_ID -o ${save_path}/dataset${DATASET_ID}_HN_fold1 -c 3d_fullres -tr nnUNetTrainerMRCT_track -f 1
# nnUNetv2_export_model_to_zip -d $DATASET_ID -o ${save_path}/dataset${DATASET_ID}_HN_fold2 -c 3d_fullres -tr nnUNetTrainerMRCT_track -f 2
# nnUNetv2_export_model_to_zip -d $DATASET_ID -o ${save_path}/dataset${DATASET_ID}_HN_fold3 -c 3d_fullres -tr nnUNetTrainerMRCT_track -f 3
# nnUNetv2_export_model_to_zip -d $DATASET_ID -o ${save_path}/dataset${DATASET_ID}_HN_fold4 -c 3d_fullres -tr nnUNetTrainerMRCT_track -f 4

