from typing import List, Type, Optional, Tuple, Union
from nnunetv2.experiment_planning.experiment_planners.default_experiment_planner import ExperimentPlanner
import os, glob

import os, glob, shutil, json
from pathlib import Path
import SimpleITK as sitk
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

# required input dataset id, 
# DATA_STRUCT:
# |-- ORIGIN
# |   |-- Dataset_MRI2CT
# |       |-- INPUT_IMAGES
# |       |   |-- PATIENT_0001.mha
# |       |-- TARGET_IMAGES
# |           |-- PATIENT_0001.mha
# |       |-- MASKS (optional)
# |           |-- PATIENT.mha (optional)
# |       |-- Labels (optional)
# |           |-- PATIENT.mha (optional)
# |-- nnUNet_raw
# |   |-- DatasetXXX_YYY
# |-- nnUNet_preprocessed
# |   |-- DatasetXXX_YYY
# |-- nnUNet_results
# |   |-- DatasetXXX_YYY

def makedirs_raw_dataset(dataset_data_path):
    
    os.makedirs(dataset_data_path, exist_ok = True)
    os.makedirs(os.path.join(dataset_data_path, 'imagesTr'), exist_ok=True)
    os.makedirs(os.path.join(dataset_data_path, 'labelsTr'), exist_ok = True)

def process_file(data_path, dataset_path, modality_suffix="_0000"):
    curr_img = sitk.ReadImage(data_path)
    filename = os.path.basename(data_path)
    if not filename.endswith(f'{modality_suffix}.mha'):
        filename = filename + f'{modality_suffix}.mha'
    sitk.WriteImage(curr_img, os.path.join(dataset_path, f'imagesTr/{filename}'))

    data = sitk.GetArrayFromImage(curr_img)
    data = np.ones_like(data)

    filename = filename.replace(modality_suffix, '')  # Remove modality suffix for masks
    label_path = os.path.join(dataset_path, f'labelsTr/{filename}')
    if not os.path.exists(label_path):
        label_img = sitk.GetImageFromArray(data)
        label_img.SetDirection(curr_img.GetDirection())
        label_img.SetOrigin(curr_img.GetOrigin())
        label_img.SetSpacing(curr_img.GetSpacing())
        sitk.WriteImage(label_img, label_path)

def create_dataset_json(num_train, preprocessing, dataset_data_path): 
    if preprocessing.lower() == 'ct':
        preprocessing = 'CT_zscore_synthrad'

    data_dataset_json = {
        "labels": {
            "label_001": "1", 
            "background": 0
        },
        "channel_names": {
            "0": preprocessing,
            # "1": config["preprocessing_mask"],
            
        },
        "numTraining": num_train,
        "file_ending": ".mha"
    }
    dump_data_datasets_path = os.path.join(dataset_data_path, 'dataset.json')
    with open(dump_data_datasets_path, 'w') as f:
        json.dump(data_dataset_json, f)

def move_preprocessed(nnunet_datas_preprocessed_dir, nnunet_targets_preprocessed_dir, folder_name): 
    list_preprocessed_datas_seg_path = sorted(glob.glob(os.path.join(nnunet_targets_preprocessed_dir, f'{folder_name}/*_seg.npy'))) # source ct images to mri seg
    list_preprocessed_targets_path = sorted(glob.glob(os.path.join(nnunet_datas_preprocessed_dir, f'{folder_name}/*.npy'))) # target ct images
    list_preprocessed_targets_path = [name for name in list_preprocessed_targets_path if '_seg' not in name]

    assert len(list_preprocessed_datas_seg_path) == len(list_preprocessed_targets_path)
    assert len(list_preprocessed_datas_seg_path) > 0, "No preprocessed data found in the specified directory."

    for (datas_path, targets_path) in zip(list_preprocessed_datas_seg_path, list_preprocessed_targets_path):
        print(targets_path, "->", datas_path)
        shutil.copy(src = targets_path, dst = datas_path) 

def move_preprocessed_mask(nnunet_datas_preprocessed_dir, nnunet_targets_preprocessed_dir, folder_name):
    list_preprocessed_targets_path = sorted(glob.glob(os.path.join(nnunet_datas_preprocessed_dir, f'{folder_name}/*_seg.npy'))) # target mask images

    assert len(list_preprocessed_targets_path) > 0, "No preprocessed data found in the specified directory."

    for targets_path in list_preprocessed_targets_path:
        datas_path = os.path.join(nnunet_targets_preprocessed_dir, folder_name, os.path.basename(targets_path).replace('_seg', '_mask'))
        print(targets_path, "->", datas_path)
        shutil.copy(src = targets_path, dst = datas_path)

def move_masks(list_data_mask, dataset_mask_path):

    def _process_mask_file(data_path, dataset_mask_path):

        filename = os.path.basename(data_path)
        shutil.copy(data_path, os.path.join(dataset_mask_path, filename))


    # Use the affine from the last MRI as a placeholder, but for sitk we use spacing/origin/direction from the image itself
    os.makedirs(dataset_mask_path, exist_ok=True) 
    with ThreadPoolExecutor() as executor:
        list(tqdm(executor.map(lambda data_path: _process_mask_file(data_path, dataset_mask_path), list_data_mask), total=len(list_data_mask)))

def move_gt_target(list_data_ct, dataset_target_path2):

    def _process_target_file(data_path, dataset_target_path2):

        filename = os.path.basename(data_path)
        shutil.copy(data_path, os.path.join(dataset_target_path2, filename))


    # Use the affine from the last MRI as a placeholder, but for sitk we use spacing/origin/direction from the image itself
    os.makedirs(dataset_target_path2, exist_ok=True) 
    with ThreadPoolExecutor() as executor:
        list(tqdm(executor.map(lambda data_path: _process_target_file(data_path, dataset_target_path2), list_data_ct), total=len(list_data_ct)))

def move_gt_plan(nnunet_datas_preprocessed_dir, dataset_target_plan_path):
    os.makedirs(dataset_target_plan_path, exist_ok=True)

    list_preprocessed_plan_path = sorted(glob.glob(os.path.join(nnunet_datas_preprocessed_dir, f'*.json'))) # copy all json files
    assert len(list_preprocessed_plan_path) == 3, "No preprocessed plan found in the target image directory."

    for json_file in list_preprocessed_plan_path:
        shutil.copy(src = json_file, dst = dataset_target_plan_path)
        print(json_file, "->", dataset_target_plan_path)


def nnsyn_plan_and_preprocess(data_origin_path: str, dataset_id: int,  
                        preprocessing_input: str, preprocessing_target: str,
                        configuration: str = '3d_fullres', planner_class: str = 'ExperimentPlanner', plan: str = 'nnUNetPlans', 
                        dataset_name: str = None, use_mask: bool = False):
    list_data_cbct = sorted(glob.glob(os.path.join(data_origin_path, 'INPUT_IMAGES','*.mha'), recursive=True))
    list_data_mask = sorted(glob.glob(os.path.join(data_origin_path, 'MASKS','*.mha'), recursive=True))
    list_data_ct = sorted(glob.glob(os.path.join(data_origin_path, 'TARGET_IMAGES','*.mha'), recursive=True))
    print("input1 ---", len(list_data_cbct), list_data_cbct[:2])
    print("input2 ---", len(list_data_mask), list_data_mask[:2])
    print("target ---", len(list_data_ct), list_data_ct[:2])

    if dataset_name is None:
        dataset_name = os.path.basename(data_origin_path)

    if len(list_data_mask) == 0:
        use_mask = False
        print("No masks found, set use_mask to False")
    else:
        use_mask = True

    # copy data from orign to nnUNet_raw
    dataset_data_path = os.path.join(os.environ['nnUNet_raw'], f'Dataset{dataset_id:03d}_{dataset_name}') 
    dataset_target_path = os.path.join(os.environ['nnUNet_raw'], f'Dataset{dataset_id+1:03d}_{dataset_name}') 
    makedirs_raw_dataset(dataset_data_path)
    makedirs_raw_dataset(dataset_target_path)

    with ThreadPoolExecutor() as executor:
        list(tqdm(executor.map(lambda data_path: process_file(data_path, dataset_data_path, "_0000"), list_data_cbct), total=len(list_data_cbct)))

    with ThreadPoolExecutor() as executor:
        list(tqdm(executor.map(lambda target_path: process_file(target_path, dataset_target_path, "_0000"), list_data_ct), total=len(list_data_ct)))

    if use_mask:
        move_masks(list_data_mask, os.path.join(dataset_data_path, 'labelsTr'))
        print("Masks moved to:", os.path.join(dataset_data_path, 'labelsTr'))

        move_masks(list_data_mask, os.path.join(dataset_target_path, 'labelsTr'))
        print("Masks moved to:", os.path.join(dataset_target_path, 'labelsTr'))
    
    # create dataset.json
    num_train = len(list_data_cbct)
    assert len(list_data_cbct) == len(list_data_ct)
    create_dataset_json(num_train, preprocessing_input, dataset_data_path)
    create_dataset_json(num_train, preprocessing_target, dataset_target_path)

    # apply preprocessing and unpacking
    if 'MPLBACKEND' in os.environ: 
        del os.environ['MPLBACKEND'] # avoid conflicts with matplotlib backend

    os.system(f'nnUNetv2_plan_and_preprocess -d {dataset_id} -c {configuration} -pl {planner_class}')
    os.system(f'nnUNetv2_unpack {dataset_id} {configuration} 0 -p {plan}')

    os.system(f'nnUNetv2_plan_and_preprocess -d {dataset_id + 1} -c {configuration} -pl {planner_class}')
    os.system(f'nnUNetv2_unpack {dataset_id + 1} {configuration} 0 -p {plan}')

    # move preprocessed targets to data
    nnunet_datas_preprocessed_dir = os.path.join(os.environ['nnUNet_preprocessed'], f'Dataset{dataset_id+1:03d}_{dataset_name}') 
    nnunet_targets_preprocessed_dir = os.path.join(os.environ['nnUNet_preprocessed'], f'Dataset{dataset_id:03d}_{dataset_name}') 
    move_preprocessed(nnunet_datas_preprocessed_dir, nnunet_targets_preprocessed_dir, f'nnUNetPlans_3d_fullres')

    if use_mask:
        move_preprocessed_mask(nnunet_datas_preprocessed_dir, nnunet_targets_preprocessed_dir, f'nnUNetPlans_3d_fullres')
        dataset_mask_path = os.path.join(os.environ['nnUNet_preprocessed'], f'Dataset{dataset_id:03d}_{dataset_name}', 'masks')
        move_masks(list_data_mask, dataset_mask_path)

    dataset_target_path2 = os.path.join(os.environ['nnUNet_preprocessed'], f'Dataset{dataset_id:03d}_{dataset_name}', 'gt_target')
    move_gt_target(list_data_ct, dataset_target_path2)

    dataset_target_plan_path = os.path.join(os.environ['nnUNet_preprocessed'], f'Dataset{dataset_id:03d}_{dataset_name}', 'gt_plan')
    move_gt_plan(nnunet_datas_preprocessed_dir, dataset_target_plan_path)

    # list_gt_target_segmentation_ts = sorted(glob.glob(os.path.join(config["data_root"], '**','segmentation_ct_stitched_resampled.mha'), recursive=True))
    # print("gt_target_segmentation_ts ---", len(list_gt_target_segmentation_ts), list_gt_target_segmentation_ts[:2])
    # dataset_target_path2 = os.path.join(os.environ['nnUNet_preprocessed'], f'Dataset{dataset_id:03d}_{dataset_name}', 'gt_target_segmentation_ts')
    # move_gt_target(list_gt_target_segmentation_ts, dataset_target_path2)

    # remove target datasets to save space
    shutil.rmtree(os.path.join(os.environ['nnUNet_raw'], f'Dataset{dataset_id+1:03d}_{dataset_name}') )
    shutil.rmtree(os.path.join(os.environ['nnUNet_preprocessed'], f'Dataset{dataset_id+1:03d}_{dataset_name}') )
    shutil.rmtree(os.path.join(os.environ['nnUNet_results'], f'Dataset{dataset_id+1:03d}_{dataset_name}') )

if __name__ == "__main__":
    # data_origin_path = '/datasets/work/hb-synthrad2023/work/synthrad2025/bw_workplace/data/nnunet_struct/ORIGIN/Dataset_MRI2CT'
    # dataset_id = 980
    # dataset_name = 'Dataset_MRI2CT_debug'
    # preprocessing_input = 'MR'
    # preprocessing_target = 'CT_zscore_synthrad'
    # use_mask = False
    # configuration = '3d_fullres'
    # planner_class = 'ExperimentPlanner'
    # plan = 'nnUNetPlans'

    # data_origin_path = '/datasets/work/hb-synthrad2023/work/synthrad2025/bw_workplace/data/nnunet_struct/ORIGIN/Dataset_MRI2CT'
    # dataset_id = 981
    # dataset_name = 'Dataset_MRI2CT_res_debug'
    # preprocessing_input = 'MR'
    # preprocessing_target = 'CT_zscore_synthrad'
    # use_mask = False
    # configuration = '3d_fullres'
    # planner_class = 'nnUNetPlannerResEncL'
    # plan = 'nnUNetResEncUNetLPlans'

    data_origin_path = '/datasets/work/hb-synthrad2023/work/synthrad2025/bw_workplace/data/nnunet_struct/ORIGIN/Synthrad2025_MRI2CT_AB'
    dataset_id = 982
    preprocessing_input = 'MR'
    preprocessing_target = 'CT'
    configuration = '3d_fullres'
    planner_class = 'nnUNetPlannerResEncL'
    plan = 'nnUNetResEncUNetLPlans'
    # dataset_name = 'Dataset_MRI2CT_res_mask_debug'
    # use_mask = False

    nnsyn_plan_and_preprocess(data_origin_path, dataset_id, preprocessing_input, preprocessing_target, configuration, planner_class, plan, use_mask=False)
    