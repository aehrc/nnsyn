import pathlib
import os, glob, shutil, json
from pathlib import Path
import SimpleITK as sitk
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import datetime
from pprint import pprint
from nnunetv2.analysis.revert_normalisation import get_ct_normalisation_values, revert_normalisation


def process_file(data_path, dataset_path, modality_suffix="_0000"):
    filename = data_path.split(os.sep)[-2]
    if not filename.endswith(f'{modality_suffix}.mha'):
        filename = filename + f'{modality_suffix}.mha'
    os.makedirs(os.path.join(dataset_path, 'imagesVal'), exist_ok=True)
    shutil.copyfile(data_path, os.path.join(dataset_path, f'imagesVal/{filename}'))

def process_region_dataset(config, submission_path):
    print(config)

    dataset_id = config['dataset_id']
    fold = config['fold']
    dataset_data_name = config['dataset_data_name']
    dataset_shortname = config['data_root'].split(os.sep)[-1]  # e.g., "synthrad2025_task1_MR_TH" -> "TH"
    dataset_data_path = os.path.join(os.environ['nnUNet_raw'], f'Dataset{dataset_id:03d}_{dataset_data_name}')
    list_data_mri = sorted(glob.glob(os.path.join(data_root, dataset_shortname, '**','mr.mha'), recursive=True))
    print(os.path.join(data_root, dataset_shortname, '**','mr.mha'))
    print("input1 ---", len(list_data_mri), list_data_mri)

    # copy validation images to imagesVal
    with ThreadPoolExecutor() as executor:
        list(tqdm(executor.map(lambda data_path: process_file(data_path, dataset_data_path, "_0000"), list_data_mri), total=len(list_data_mri)))

    # predict images
    input_dir = os.path.join(dataset_data_path, 'imagesVal')
    output_dir = os.path.join(submission_path, f'DatasetID{dataset_id:03d}_{dataset_data_name}')
    os.makedirs(output_dir, exist_ok=True)
    os.system(f"nnUNetv2_predict -d {dataset_id} -i {input_dir} -o {output_dir} -c {config['configuration']} -p {config['plan']} -tr {config['trainer']} -f {fold}")

    # revert normalisation
    print("Reverting normalisation...")
    ct_plan_path = f"/datasets/work/hb-synthrad2023/work/synthrad2025/bw_workplace/data/nnunet_struct/preprocessed/Dataset{config['dataset_id'] + 1}_{config['dataset_target_name']}/nnUNetPlans.json"
    ct_mean, ct_std = get_ct_normalisation_values(ct_plan_path)
    revert_normalisation(output_dir, ct_mean, ct_std, save_path=output_dir + "_revert_norm")
    # move previous results to backup
    if os.path.exists(output_dir):
        shutil.move(output_dir, output_dir + "_revert_norm/backup_normalised")

def rename_and_archive(submission_path):
    submission_name = submission_path.split(os.sep)[-1]
    print("Finalizing submission...")
    save_path = os.path.join(submission_path, 'Folder')
    os.makedirs(save_path, exist_ok=True)

    for file in tqdm(glob.glob(os.path.join(submission_path, "*_revert_norm", "*.mha"))):
        new_file = os.path.join(save_path, "sct_" + os.path.basename(file))
        shutil.copyfile(file, new_file)
    print("Files copied to submission folder:", len(os.listdir(save_path)))
    # zip folder
    shutil.make_archive(save_path, 'zip', save_path)
    print("Copying the line below to the local machine to download the zip file:")
    print(f'rsync -avz xin015@virga.hpc.csiro.au:{save_path}.zip ./synthrad2025_submission/{submission_name}/')



if __name__ == "__main__":

    # nnunet environment variables
    os.environ["nnUNet_raw"] = "/datasets/work/hb-synthrad2023/work/synthrad2025/bw_workplace/data/nnunet_struct/raw"
    os.environ["nnUNet_preprocessed"] = "/datasets/work/hb-synthrad2023/work/synthrad2025/bw_workplace/data/nnunet_struct/preprocessed"
    os.environ["nnUNet_results"] = "/datasets/work/hb-synthrad2023/work/synthrad2025/bw_workplace/data/nnunet_struct/results"

    # important paths
    submission_name = f'task1_{datetime.datetime.now().strftime("%Y%m%d")}_3models_v2r_deformed_ants_full' # need to rename

    data_root = "/datasets/work/hb-synthrad2023/source/synthrad2025_data_v2r/synthRAD2025_Task1_Val/Task1"
    submission_root = "/datasets/work/hb-synthrad2023/work/synthrad2025/bw_workplace/data/nnunet_struct/submission"
    submission_path = os.path.join(submission_root, submission_name)

    # predict
    cur_file_path = pathlib.Path(__file__).parent.resolve()
    config_AB = json.load(open(os.path.join(cur_file_path, "config_270__nnUNetTrainerMRCT__nnUNetPlans__3d_fullres.json")))
    process_region_dataset(config_AB, submission_path)

    config_HN = json.load(open(os.path.join(cur_file_path, "config_272__nnUNetTrainerMRCT__nnUNetPlans__3d_fullres.json")))
    process_region_dataset(config_HN, submission_path)

    config_TH = json.load(open(os.path.join(cur_file_path, "config_274__nnUNetTrainerMRCT__nnUNetPlans__3d_fullres.json")))
    process_region_dataset(config_TH, submission_path)

    # final archive and submission
    rename_and_archive(submission_path)
    