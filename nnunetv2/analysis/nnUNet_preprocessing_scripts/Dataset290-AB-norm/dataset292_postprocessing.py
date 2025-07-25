import sys
import json
import os
sys.path.append("/datasets/work/hb-synthrad2023/work/synthrad2025/bw_workplace/ref/nnUNet_translation/nnunetv2/analysis")
from revert_normalisation import get_ct_normalisation_values, revert_normalisation
from result_analysis import TestingResults

# Load config file
os.environ_path = "/datasets/work/hb-synthrad2023/work/synthrad2025/bw_workplace/data/nnunet_struct/raw"
nnUNet_preprocessed_path = "/datasets/work/hb-synthrad2023/work/synthrad2025/bw_workplace/data/nnunet_struct/preprocessed"
nnUNet_results_path = "/datasets/work/hb-synthrad2023/work/synthrad2025/bw_workplace/data/nnunet_struct/results"

config = json.load(open("/datasets/work/hb-synthrad2023/work/synthrad2025/bw_workplace/ref/nnUNet_translation/nnunetv2/analysis/nnUNet_preprocessing_scripts/Dataset290-AB-norm/config_290.json"))
plan = "nnUNetPlans"
trainer = "nnUNetTrainerMRCT"
configuration = "3d_fullres"
fold = 0


ct_plan_path = f"{nnUNet_preprocessed_path}/Dataset{config['dataset_id'] + 1}_{config['dataset_target_name']}/{plan}.json"
ct_mean, ct_std = get_ct_normalisation_values(ct_plan_path)

pred_path = f"{nnUNet_results_path}/Dataset{config['dataset_id']}_{config['dataset_data_name']}/{trainer}__{plan}__{configuration}/fold_{fold}/validation"
pred_path_revert_norm = pred_path + "_revert_norm"
revert_normalisation(pred_path, ct_mean, ct_std, save_path=pred_path_revert_norm)


ts = TestingResults(pred_path_revert_norm, task=1, region='AB')
ts.process_patients()

print('End')
