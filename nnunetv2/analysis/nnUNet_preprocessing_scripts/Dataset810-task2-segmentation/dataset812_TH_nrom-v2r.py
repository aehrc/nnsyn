import os, glob
import sys
sys.path.append('/datasets/work/hb-synthrad2023/work/synthrad2025/bw_workplace/ref/nnUNet_translation/nnunetv2/analysis')
from organise_dataset_segmenation import *
import pathlib

if __name__ == '__main__':


    TASK = 2
    REGION = "TH"
    DATASET_ID = 812

    SOURCE_DATASET_ID = 545 # synthetic CT dataset
    config = {
        "dataset_id": DATASET_ID,  # Updated to 200 for CT noNorm
        "dataset_data_name": f"SEGMENTATION_synthrad2025_task{TASK}_CT_{REGION}_aligned_to_Dataset{SOURCE_DATASET_ID}",
        "data_root": f"/datasets/work/hb-synthrad2023/source/synthrad2025_data_v2r/synthRAD2025_Task{TASK}_Train/Task{TASK}/{REGION}", # include centreD
        "preprocessing_CT": "CT_zscore_synthrad", 
        "preprocessing_mask": "masked",
        "fold": 0,
        "configuration": "3d_fullres",
        "trainer": "nnUNetTrainer",
        "planner_class": "nnUNetPlannerResEncL",
        "plan": "nnUNetResEncUNetLPlans", 
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
    list_data_segmentation = sorted(glob.glob(os.path.join(config["data_root"], '**','segmentation_ct_stitched_resampled.mha'), recursive=True))
    list_data_ct = sorted(glob.glob(os.path.join(config["data_root"], '**','ct_stitched_resampled.mha'), recursive=True))
    print("input2 ---", len(list_data_segmentation), list_data_segmentation[:2])
    print("target ---", len(list_data_ct), list_data_ct[:2])

    ## Define dataset ID and make paths
    dataset_id = config["dataset_id"]
    dataset_data_name = config["dataset_data_name"]

    # we will copy the datas
    # do not use exist_ok=True, we want an error if the dataset exist already
    dataset_data_path = os.path.join(os.environ['nnUNet_raw'], f'Dataset{dataset_id:03d}_{dataset_data_name}') 
    makedirs_raw_dataset(dataset_data_path)

    
    labels_to_use = get_classes_to_use(config["region"])
    
    with ThreadPoolExecutor() as executor:
        list(tqdm(executor.map(lambda target_path: process_ct_file(target_path, dataset_data_path, "_0000"), list_data_ct), total=len(list_data_ct)))

    with ThreadPoolExecutor() as executor:
        list(tqdm(executor.map(lambda data_path: process_segmentation_file(data_path, dataset_data_path, labels_to_use), list_data_segmentation), total=len(list_data_segmentation)))

    num_train = len(list_data_ct)
    preproceessing = config["preprocessing_CT"]
    create_dataset_json(num_train, preproceessing, dataset_data_path, labels_to_use)

    # move plans between datasets
    
    SOURCE_PLAN_IDENTIFIER = config["plan"]
    TARGET_PLAN_IDENTIFIER = config["plan"] + f'_Dataset{SOURCE_DATASET_ID}'

    os.system(f'nnUNetv2_extract_fingerprint -d {dataset_id} --verify_dataset_integrity')
    os.system(f'nnUNetv2_move_plans_between_datasets -s {SOURCE_DATASET_ID} -t {dataset_id} -sp {config["plan"]} -tp {TARGET_PLAN_IDENTIFIER}')
    copy_dataset_json(os.environ['nnUNet_raw'], os.environ['nnUNet_preprocessed'], f'Dataset{dataset_id:03d}_{dataset_data_name}')
    os.system(f'nnUNetv2_preprocess -d {dataset_id} -c {config["configuration"]} -plans_name {TARGET_PLAN_IDENTIFIER} -np 4')
    # os.system(f'nnUNetv2_plan_and_preprocess -d {dataset_id} -c {config["configuration"]} -pl {config["planner_class"]}')
    # Train the model
    # os.system(f'nnUNetv2_train {dataset_id} {config["configuration"]} {config["fold"]} -tr {config["trainer"]}')