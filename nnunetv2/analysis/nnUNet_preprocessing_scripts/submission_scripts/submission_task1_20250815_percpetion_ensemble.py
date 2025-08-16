import os, glob, shutil, json
from pathlib import Path
import SimpleITK as sitk
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import datetime
from pprint import pprint
from nnunetv2.analysis.revert_normalisation import get_ct_normalisation_values, revert_normalisation
import pathlib



def _process_file(data_path, dataset_path, modality_suffix="_0000"):
    filename = data_path.split(os.sep)[-2]
    if not filename.endswith(f'{modality_suffix}.mha'):
        filename = filename + f'{modality_suffix}.mha'
    os.makedirs(os.path.join(dataset_path, 'imagesVal'), exist_ok=True)
    shutil.copyfile(data_path, os.path.join(dataset_path, f'imagesVal/{filename}'))

def _process_mask_file(data_path, dataset_path):
    filename = data_path.split(os.sep)[-2]
    os.makedirs(os.path.join(dataset_path, 'mask'), exist_ok=True)
    shutil.copyfile(data_path, os.path.join(dataset_path, f'mask/{filename}.mha'))

def prepare_validation_dataset(data_root, dataset_name, task=1):
    '''
    dataset_name: e.g., "Dataset264_synthrad2025_task1_MR_TH_pre_v2r_stitched_masked"
    '''
    dataset_data_path = os.path.join(os.environ['nnUNet_raw'], dataset_name)
    input_dir = os.path.join(dataset_data_path, 'imagesVal')
    if os.path.exists(input_dir) and len(os.listdir(input_dir)) > 0:
        print(f"Input directory {input_dir} already exists. Skipping dataset preparation.")
        return
    else:
        os.makedirs(input_dir, exist_ok=True)
        print(f"Extracting data from {data_root}...")
        print(f"Creating input directory {input_dir}...")

    region = data_root.split(os.sep)[-1]  # e.g., "synthrad2025_task1_MR_TH" -> "TH"
    if '_AB_' in dataset_name:
        region = 'AB'
    elif '_TH_' in dataset_name:
        region = 'TH'
    elif '_HN_' in dataset_name:
        region = 'HN'
    else:
        raise ValueError(f"Unknown region in dataset name: {dataset_name}")
    input_img_path = os.path.join(data_root, region, '**','mr.mha') if task == 1 else os.path.join(data_root, region, '**','cbct.mha')
    list_data_mri = sorted(glob.glob(input_img_path, recursive=True))
    list_data_mask = sorted(glob.glob(os.path.join(data_root, region, '**','mask.mha'), recursive=True))
    print("input1 ---", len(list_data_mri), list_data_mri)
    print("input2 ---", len(list_data_mask), list_data_mask)
    

    # copy validation images to imagesVal
    with ThreadPoolExecutor() as executor:
        list(tqdm(executor.map(lambda data_path: _process_file(data_path, dataset_data_path, "_0000"), list_data_mri), total=len(list_data_mri)))

    # copy validation masks to masks
    with ThreadPoolExecutor() as executor:
        list(tqdm(executor.map(lambda data_path: _process_mask_file(data_path, dataset_data_path), list_data_mask), total=len(list_data_mask)))

    # predict images
    

def predict_and_revert(dataset_name, method_folder_name, submission_path, fold=0, chk="checkpoint_final.pth"):
    '''
    method_folder_name: e.g., "nnUNetTrainerMRCT_loss_masked_perception_masked__nnUNetResEncUNetLPlans__3d_fullres"
    '''

    trainer, plan_name, configuration = method_folder_name.split('__')

    input_dir = os.path.join(os.environ['nnUNet_raw'], dataset_name, 'imagesVal')
    output_dir = os.path.join(submission_path, submission_name, dataset_name)
    os.makedirs(output_dir, exist_ok=True)
    os.system(f"nnUNetv2_predict -d {dataset_name} -i {input_dir} -o {output_dir} -c {configuration} -p {plan_name} -tr {trainer} -f {fold} -chk {chk}")

    # revert normalisation
    print("Reverting normalisation...")
    mask_dir = os.path.join(os.environ['nnUNet_raw'], dataset_name, 'mask')
    ct_plan_path = os.path.join(os.environ["nnUNet_preprocessed"], dataset_name, f"gt_{plan_name}.json")
    ct_mean, ct_std = get_ct_normalisation_values(ct_plan_path)
    revert_normalisation(output_dir, ct_mean, ct_std, save_path=output_dir + "_revert_norm", \
                         mask_path=mask_dir, mask_outside_value=-1000)
    # move previous results to backup
    if os.path.exists(output_dir):
        shutil.move(output_dir, output_dir + "_revert_norm/backup_normalised")

def export_checkpoints(dataset_name, method_folder_name, submission_path, fold=0):
    """
    Export the model checkpoints to the submission folder.
    """
    trainer, plan_name, configuration = method_folder_name.split('__')
    save_path = os.path.join(submission_path, submission_name, dataset_name + "_revert_norm", f'{dataset_name.split("_")[0]}_{method_folder_name}_fold{fold}.zip')
    command = f"nnUNetv2_export_model_to_zip -d {dataset_name} -o {save_path} -c {configuration} -tr {trainer} -p {plan_name} -f {fold}"
    print(command)
    os.system(command)

if __name__ == "__main__":

    # nnunet environment variables
    os.environ["nnUNet_raw"] = "/datasets/work/hb-synthrad2023/work/synthrad2025/bw_workplace/data/nnunet_struct/raw"
    os.environ["nnUNet_preprocessed"] = "/datasets/work/hb-synthrad2023/work/synthrad2025/bw_workplace/data/nnunet_struct/preprocessed"
    os.environ["nnUNet_results"] = "/datasets/work/hb-synthrad2023/work/synthrad2025/bw_workplace/data/nnunet_struct/results"

    # important paths
    TASK = 1
    DESCRIPTION = "perception_masked_ensemble"
    DATASET_NAME_AB = "Dataset260_synthrad2025_task1_MR_AB_pre_v2r_stitched_masked"
    DATASET_NAME_HN = "Dataset262_synthrad2025_task1_MR_HN_pre_v2r_stitched_masked"
    DATASET_NAME_TH = "Dataset264_synthrad2025_task1_MR_TH_pre_v2r_stitched_masked"
    METHOD_FOLDER_AB = "nnUNetTrainerMRCT_loss_masked_perception_masked__nnUNetResEncUNetLPlans__3d_fullres"
    METHOD_FOLDER_HN = "nnUNetTrainerMRCT_loss_masked_perception_masked__nnUNetResEncUNetLPlans__3d_fullres"
    METHOD_FOLDER_TH = "nnUNetTrainerMRCT_loss_masked_perception_masked__nnUNetResEncUNetLPlans__3d_fullres"
    FOLD_AB = "0 1 2 3 4"  # folds to use for AB region
    FOLD_HN = "0 1 2 3 4"  # folds to use for HN region
    FOLD_TH = "0 1 2 3 4"  # folds to use for TH region
    # FOLD_AB = 0
    # FOLD_HN = 0
    # FOLD_TH = 0

    data_root = f"/datasets/work/hb-synthrad2023/source/synthrad2025_data_v2r/synthRAD2025_Task{TASK}_Val/Task{TASK}"
    submission_name = f'task{TASK}_{datetime.datetime.now().strftime("%Y%m%d")}_{DESCRIPTION}' # need to rename


    submission_root = "/datasets/work/hb-synthrad2023/work/synthrad2025/bw_workplace/data/nnunet_struct/submission"
    submission_path = os.path.join(submission_root, submission_name)
    os.makedirs(submission_path, exist_ok=True)


    # generate validation datasets
    ##### CHANGE DATASET NAMES HERE #####
    # print("Preparing validation datasets...")
    # prepare_validation_dataset(data_root, DATASET_NAME_AB, task=TASK)
    # prepare_validation_dataset(data_root, DATASET_NAME_HN, task=TASK)
    # prepare_validation_dataset(data_root, DATASET_NAME_TH, task=TASK)

    # # predict
    print("Predicting and reverting normalisation for AB region...")
    predict_and_revert(DATASET_NAME_AB, METHOD_FOLDER_AB, submission_path, fold=FOLD_AB)

    print("Predicting and reverting normalisation for HN region...")
    predict_and_revert(DATASET_NAME_HN, METHOD_FOLDER_HN, submission_path, fold=FOLD_HN)

    print("Predicting and reverting normalisation for TH region...")
    predict_and_revert(DATASET_NAME_TH, METHOD_FOLDER_TH, submission_path, fold=FOLD_TH)

    # export checkpoints to submission folder
    print("Exporting checkpoints to submission folder...")
    export_checkpoints(DATASET_NAME_AB, METHOD_FOLDER_AB, submission_path, fold=FOLD_AB)
    export_checkpoints(DATASET_NAME_HN, METHOD_FOLDER_HN, submission_path, fold=FOLD_HN)
    export_checkpoints(DATASET_NAME_TH, METHOD_FOLDER_TH, submission_path, fold=FOLD_TH)

    # final archive and submission
    print("Finalizing submission...")
    save_path = os.path.join(submission_path, submission_name, 'Folder')
    os.makedirs(save_path, exist_ok=True)

    for file in glob.glob(os.path.join(submission_path, submission_name, "*_revert_norm", "*.mha")):
        new_file = os.path.join(save_path, "sct_" + os.path.basename(file))
        shutil.copyfile(file, new_file)
    # zip folder
    shutil.make_archive(save_path, 'zip', save_path)
    print("Copying the line below to the local machine to download the zip file:")
    print('cd ~/Downloads')
    print(f'rsync -avz xin015@virga.hpc.csiro.au:{save_path}.zip ./synthrad2025_submission/{submission_name}/')

