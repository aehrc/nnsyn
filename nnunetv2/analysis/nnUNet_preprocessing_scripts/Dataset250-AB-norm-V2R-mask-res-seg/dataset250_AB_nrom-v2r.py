import os, glob
import sys
sys.path.append('/datasets/work/hb-synthrad2023/work/synthrad2025/bw_workplace/ref/nnUNet_translation/nnunetv2/analysis')
from organise_dataset import *
import pathlib

if __name__ == '__main__':


    TASK = 1
    REGION = "AB"
    DATASET_ID = 250
    # total segmentation results saved to gt_target_segmentation
    # our unet segmentation results saved to mask

    # THIS IS REGION SPECIFIC. NEED TO CHANGE
    segmentation_gt_ours = {
        "AB": "/datasets/work/hb-synthrad2023/work/synthrad2025/bw_workplace/data/nnunet_struct/raw/Dataset800_SEGMENTATION_synthrad2025_task1_CT_AB_aligned_to_Dataset261/labelsTr",
        "HN": "/datasets/work/hb-synthrad2023/work/synthrad2025/bw_workplace/data/nnunet_struct/raw/Dataset801_SEGMENTATION_synthrad2025_task1_CT_HN_aligned_to_Dataset263/labelsTr",
        "TH": "/datasets/work/hb-synthrad2023/work/synthrad2025/bw_workplace/data/nnunet_struct/raw/Dataset802_SEGMENTATION_synthrad2025_task1_CT_TH_aligned_to_Dataset265/labelsTr",
    }

    SEGMENTATION_GT_OURS_FOLDER = segmentation_gt_ours[REGION]
    config = {
        "dataset_id": DATASET_ID,  # Updated to 200 for CT noNorm
        "dataset_data_name": f"synthrad2025_task1_MR_{REGION}_pre_v2r_stitched_masked_synseg",
        "dataset_target_name": f"synthrad2025_task1_CT_{REGION}_pre_v2r_stitched_masked_synseg",
        "data_root": f"/datasets/work/hb-synthrad2023/source/synthrad2025_data_v2r/synthRAD2025_Task1_Train/Task1/{REGION}", # include centreD
        "preprocessing_CT": "CT_zscore_synthrad", 
        "preprocessing_MRI": "MR",
        "preprocessing_mask": "masked_segmentation",
        "fold": 0,
        "configuration": "3d_fullres",
        "trainer": "nnUNetTrainerMRCT_loss_masked",
        "planner_class": "nnUNetPlannerResEncL",
        "plan": "nnUNetResEncUNetLPlans", 
        "task": TASK,
        "region": REGION,
        "segmentation_gt_ours": SEGMENTATION_GT_OURS_FOLDER,
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
    list_data_ct = sorted(glob.glob(os.path.join(config["data_root"], '**','ct_stitched_resampled.mha'), recursive=True))
    list_gt_target_segmentation_ts = sorted(glob.glob(os.path.join(config["data_root"], '**','segmentation_ct_stitched_resampled.mha'), recursive=True))
    list_gt_target_segmentation_ours = sorted(glob.glob(os.path.join(config["data_root"], '**','segmentation_masked_ct_stitched_resampled.mha'), recursive=True))
    list_data_mask = sorted(glob.glob(os.path.join(config["data_root"], '**','mask.mha'), recursive=True))
    
    print("input1 ---", len(list_data_mri), list_data_mri[:2])
    print("target ---", len(list_data_ct), list_data_ct[:2])
    print("segmentation ---", len(list_gt_target_segmentation_ts), list_gt_target_segmentation_ts[:2])
    print("segmentation ---", len(list_gt_target_segmentation_ours), list_gt_target_segmentation_ours[:2])
    print("mask ---", len(list_data_mask), list_data_mask[:2])


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

    with ThreadPoolExecutor() as executor:
        list(tqdm(executor.map(lambda data_path: process_file(data_path, dataset_data_path, "_0000", 0), list_data_mri), total=len(list_data_mri)))

    with ThreadPoolExecutor() as executor:
        list(tqdm(executor.map(lambda target_path: process_file(target_path, dataset_target_path, "_0000", -1000), list_data_ct), total=len(list_data_ct)))

    labels_to_use = get_classes_to_use(config["region"])
    with ThreadPoolExecutor() as executor:
        list(tqdm(executor.map(lambda data_path: process_segmentation_file(data_path, dataset_target_path, labels_to_use), list_gt_target_segmentation_ours), total=len(list_gt_target_segmentation_ours)))

    # move our segmentation ground truth to the target dataset
    print("Segmentation moved to:", os.path.join(dataset_target_path, 'labelsTr'))
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
        
    os.system(f'nnUNetv2_plan_and_preprocess -d {dataset_id} -c {config["configuration"]} -pl {config["planner_class"]}')
    os.system(f'nnUNetv2_unpack {dataset_id} {config["configuration"]} {config["fold"]} -p {config["plan"]}')

    os.system(f'nnUNetv2_plan_and_preprocess -d {dataset_id + 1} -c {config["configuration"]} -pl {config["planner_class"]}')
    os.system(f'nnUNetv2_unpack {dataset_id + 1} {config["configuration"]} {config["fold"]} -p {config["plan"]}')


    # move preprocessed targets to data

    nnunet_datas_preprocessed_dir = os.path.join(os.environ['nnUNet_preprocessed'], f'Dataset{dataset_id+1:03d}_{dataset_target_name}') 
    nnunet_targets_preprocessed_dir = os.path.join(os.environ['nnUNet_preprocessed'], f'Dataset{dataset_id:03d}_{dataset_data_name}') 
    move_preprocessed(nnunet_datas_preprocessed_dir, nnunet_targets_preprocessed_dir, f'nnUNetPlans_3d_fullres')
    move_preprocessed_mask(nnunet_datas_preprocessed_dir, nnunet_targets_preprocessed_dir, f'nnUNetPlans_3d_fullres')
    
    dataset_mask_path = os.path.join(os.environ['nnUNet_preprocessed'], f'Dataset{dataset_id:03d}_{dataset_data_name}', 'masks')
    shutil.copytree(os.path.join(dataset_target_path, 'labelsTr'), dataset_mask_path, dirs_exist_ok=True)
    
    dataset_target_path2 = os.path.join(os.environ['nnUNet_preprocessed'], f'Dataset{dataset_id:03d}_{dataset_data_name}', 'gt_target')
    move_gt_target(list_data_ct, dataset_target_path2)

    # MOVE TOTAL SEGMENTATOR GT 
    dataset_target_path2 = os.path.join(os.environ['nnUNet_preprocessed'], f'Dataset{dataset_id:03d}_{dataset_data_name}', 'gt_target_segmentation_ts')
    move_gt_target(list_gt_target_segmentation_ts, dataset_target_path2)

    # MOVE REAL MASKS
    dataset_mask_path_real = os.path.join(os.environ['nnUNet_preprocessed'], f'Dataset{dataset_id:03d}_{dataset_data_name}', 'masks_real')
    move_gt_target(list_data_mask, dataset_mask_path_real)
    # train the network
    # os.system(f'nnUNetv2_train {dataset_id} {config["configuration"]} {config["fold"]} -tr {config["trainer"]}')