import pathlib
import sys
import json
import os
sys.path.append("/datasets/work/hb-synthrad2023/work/synthrad2025/bw_workplace/ref/nnUNet_translation/nnunetv2/analysis")
from revert_normalisation import get_ct_normalisation_values, revert_normalisation
from result_analysis import FinalValidationResults
from os.path import join

# Load config file
def analyze_validation(config):
    plan = config['plan']
    trainer = config['trainer']
    configuration = config['configuration']
    fold = config['fold']

    nnUNet_raw_path = config['nnUNet_raw']
    nnUNet_preprocessed_path = config['nnUNet_preprocessed']
    nnUNet_results_path = config['nnUNet_results']


    ct_plan_path = f"{nnUNet_preprocessed_path}/Dataset{config['dataset_id'] + 1}_{config['dataset_target_name']}/{plan}.json"
    ct_mean, ct_std = get_ct_normalisation_values(ct_plan_path)

    pred_path = f"{nnUNet_results_path}/Dataset{config['dataset_id']}_{config['dataset_data_name']}/{trainer}__{plan}__{configuration}/fold_{fold}/validation"
    pred_path_revert_norm = pred_path + "_revert_norm"
    revert_normalisation(pred_path, ct_mean, ct_std, save_path=pred_path_revert_norm)

    gt_path = join(nnUNet_preprocessed_path, f"Dataset{config['dataset_id']}_{config['dataset_data_name']}", 'gt_target')
    mask_path = join(nnUNet_preprocessed_path, f"Dataset{config['dataset_id']}_{config['dataset_data_name']}", 'masks')
    src_path = join(nnUNet_raw_path, f"Dataset{config['dataset_id']}_{config['dataset_data_name']}", 'imagesTr')
    ts = FinalValidationResults(pred_path_revert_norm, gt_path, mask_path, src_path=src_path)
    dict_metric = ts.process_patients()
    
    # raw_image_path = f"/datasets/work/hb-synthrad2023/source/synthrad2025_data_v2r/synthRAD2025_Task{task}_Train/Task{task}/{region}"
    # ts = TestingResults(pred_path_revert_norm, raw_image_path)
    # ts.process_patients()

    print('End')


if __name__ == '__main__':
    cur_file_path = pathlib.Path(__file__).parent.resolve()
    # config = json.load(open("./config_280__nnUNetTrainerMRCT__nnUNetPlans__3d_fullres.json"))
    # analyze_validation(config)
    # config = json.load(open("./config_282__nnUNetTrainerMRCT__nnUNetPlans__3d_fullres.json"))
    # analyze_validation(config)
    config = json.load(open(os.path.join(cur_file_path, "config_284__nnUNetTrainerMRCT__nnUNetPlans__3d_fullres.json")))
    analyze_validation(config)

