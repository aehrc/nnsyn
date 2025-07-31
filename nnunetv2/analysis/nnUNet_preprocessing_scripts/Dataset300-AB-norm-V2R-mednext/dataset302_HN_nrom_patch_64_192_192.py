import os, glob
import sys
sys.path.append('/datasets/work/hb-synthrad2023/work/synthrad2025/bw_workplace/ref/nnUNet_translation/nnunetv2/analysis')
from organise_dataset import *
import pathlib

if __name__ == '__main__':


    TASK = 1
    REGION = "HN"
    config = {
        "dataset_id": 302,  # Updated to 200 for CT noNorm
        "dataset_data_name": f"synthrad2025_task1_MR_{REGION}_pre_v2r_stitched",
        "dataset_target_name": f"synthrad2025_task1_CT_{REGION}_pre_v2r_stitched",
        "data_root": f"/datasets/work/hb-synthrad2023/source/synthrad2025_data_v2r/synthRAD2025_Task1_Train/Task1/{REGION}", # include centreD
        "preprocessing_CT": "CT_zscore_synthrad", 
        "preprocessing_MRI": "MR",
        "preprocessing_mask": "masked",
        "fold": 0,
        "configuration": "3d_fullres_patch_64_192_192",
        "trainer": "nnUNetTrainerV2_MedNeXt_L_kernel5",
        "plan": "nnUNetPlans", 
        "task": TASK,
        "region": REGION
    }


    # save json
    config_name = f'config_{config["dataset_id"]}__{config["trainer"]}__{config["plan"]}__{config["configuration"]}.json'
    cur_file_path = pathlib.Path(__file__).parent.resolve()
    config['save_path'] = os.path.join(cur_file_path, config_name)
    save_json(config)

    # getting input images
    nnunet_root = "/datasets/work/hb-synthrad2023/work/synthrad2025/bw_workplace/data/nnunet_struct"
    set_nnunet_path(nnunet_root, config=config)

    # example with 2 input modalities
    list_data_mri = sorted(glob.glob(os.path.join(config["data_root"], '**','mr.mha'), recursive=True))
    list_data_mask = sorted(glob.glob(os.path.join(config["data_root"], '**','mask.mha'), recursive=True))
    list_data_ct = sorted(glob.glob(os.path.join(config["data_root"], '**','ct_stitched_resampled.mha'), recursive=True))
    print("input1 ---", len(list_data_mri), list_data_mri[:3])
    print("input2 ---", len(list_data_mask), list_data_mask[:3])
    print("target ---", len(list_data_ct), list_data_ct[:3])

    ## Define dataset ID and make paths
    dataset_id = config["dataset_id"]
    dataset_data_name = config["dataset_data_name"]
    dataset_target_name = config["dataset_target_name"]

    # we will copy the datas
    # do not use exist_ok=True, we want an error if the dataset exist already
    dataset_data_path = os.path.join(os.environ['nnUNet_raw'], f'Dataset{dataset_id:03d}_{dataset_data_name}') 
    dataset_target_path = os.path.join(os.environ['nnUNet_raw'], f'Dataset{dataset_id+1:03d}_{dataset_target_name}') 
    makedirs_raw_dataset(dataset_data_path)
    makedirs_raw_dataset(dataset_target_path)

    # with ThreadPoolExecutor() as executor:
    #     list(tqdm(executor.map(lambda data_path: process_file(data_path, dataset_data_path, "_0000", 0), list_data_mri), total=len(list_data_mri)))

    # with ThreadPoolExecutor() as executor:
    #     list(tqdm(executor.map(lambda target_path: process_file(target_path, dataset_target_path, "_0000", -1000), list_data_ct), total=len(list_data_ct)))


    # create dataset.json
    # /!\ you will need to edit this with regards to the number of modalities used;
    num_train = len(list_data_mri)
    assert len(list_data_mri) == len(list_data_ct)
    config['num_train'] = num_train
    create_dataset_json(config, config["preprocessing_MRI"], dataset_data_path)
    create_dataset_json(config, config["preprocessing_CT"], dataset_target_path)




    # apply preprocessing and unpacking
    if 'MPLBACKEND' in os.environ: 
        del os.environ['MPLBACKEND'] # avoid conflicts with matplotlib backend  

    # create a new configuration for the patch size
    old_config_name = "3d_fullres"
    new_config_name = "3d_fullres_patch_64_192_192"
    attribute_to_change = "patch_size"
    new_value = [64, 192, 192]


    # os.system(f'nnUNetv2_extract_fingerprint -d {dataset_id} -c {old_config_name}')
    # os.system(f'nnUNetv2_plan_experiment -d {dataset_id} -c {old_config_name}')
    # plan_file_path = os.path.join(os.environ['nnUNet_preprocessed'], f'Dataset{dataset_id:03d}_{dataset_data_name}', f"{config['plan']}.json")
    # modify_plan(plan_file_path, old_config_name, new_config_name, attribute_to_change, new_value)
    # os.system(f'nnUNetv2_preprocess -d {dataset_id} -c {new_config_name} -np 4')
    # os.system(f'nnUNetv2_unpack {dataset_id} {new_config_name} {config["fold"]}')

    # os.system(f'nnUNetv2_extract_fingerprint -d {dataset_id + 1} -c {old_config_name}')
    # os.system(f'nnUNetv2_plan_experiment -d {dataset_id + 1} -c {old_config_name}')
    # plan_file_path = os.path.join(os.environ['nnUNet_preprocessed'], f'Dataset{dataset_id+1:03d}_{dataset_target_name}', f"{config['plan']}.json")
    # modify_plan(plan_file_path, old_config_name, new_config_name, attribute_to_change, new_value)
    # os.system(f'nnUNetv2_preprocess -d {dataset_id + 1} -c {new_config_name} -np 4')
    # os.system(f'nnUNetv2_unpack {dataset_id + 1} {new_config_name} {config["fold"]}')

    # move preprocessed targets to data

    nnunet_datas_preprocessed_dir = os.path.join(os.environ['nnUNet_preprocessed'], f'Dataset{dataset_id+1:03d}_{dataset_target_name}') 
    nnunet_targets_preprocessed_dir = os.path.join(os.environ['nnUNet_preprocessed'], f'Dataset{dataset_id:03d}_{dataset_data_name}') 
    move_preprocessed(nnunet_datas_preprocessed_dir, nnunet_targets_preprocessed_dir, folder_name=f'3d_fullres_patch_64_192_192')

    dataset_mask_path = os.path.join(os.environ['nnUNet_preprocessed'], f'Dataset{dataset_id:03d}_{dataset_data_name}', 'masks')
    move_masks(list_data_mask, dataset_mask_path)
    dataset_target_path2 = os.path.join(os.environ['nnUNet_preprocessed'], f'Dataset{dataset_id:03d}_{dataset_data_name}', 'gt_target')
    move_gt_target(list_data_ct, dataset_target_path2)

    # train the network
    # os.system(f'nnUNetv2_train {dataset_id} {config["configuration"]} {config["fold"]} -tr {config["trainer"]}')