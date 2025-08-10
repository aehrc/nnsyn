import os, glob, shutil, json
from pathlib import Path
import SimpleITK as sitk
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

def save_json(config, config_path=None):
    # save config to a JSON file
    if not config_path:
        config_path = Path(config['save_path'])
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)


def set_nnunet_path(nnunet_root, config=None):
    os.environ["nnUNet_raw"] = f"{nnunet_root}/raw"
    os.environ["nnUNet_preprocessed"] = f"{nnunet_root}/preprocessed"
    os.environ["nnUNet_results"] = f"{nnunet_root}/results"
    if config:
        config["nnUNet_raw"] = os.environ["nnUNet_raw"]
        config["nnUNet_preprocessed"] = os.environ["nnUNet_preprocessed"]
        config["nnUNet_results"] = os.environ["nnUNet_results"]
        save_json(config)



def makedirs_raw_dataset(dataset_data_path):
    
    os.makedirs(dataset_data_path, exist_ok = True)
    os.makedirs(os.path.join(dataset_data_path, 'imagesTr'), exist_ok=True)
    os.makedirs(os.path.join(dataset_data_path, 'labelsTr'), exist_ok = True)

def get_classes_to_use(region:str):
    classes_to_use = {
            "AB": [
                2, # kidney right
                3, # kidney left
                5, # liver
                6, # stomach
                *range(10, 14+1), #lungs
                *range(26, 50+1), #vertebrae
                51, #heart
                79, # spinal cord
                *range(92, 115+1), # ribs
                116 #sternum
            ],
            "HN": [
                15, # esophagus
                16, # trachea
                17, # thyroid
                *range(26, 50+1), #vertebrae
                79, #spinal cord
                90, # brain
                91, # skull
            ],
            "TH": [
                2, # kidney right
                3, # kidney left
                5, # liver
                6, # stomach
                *range(10, 14+1), #lungs
                *range(26, 50+1), #vertebrae
                51, #heart
                79, # spinal cord
                *range(92, 115+1), # ribs
                116 #sternum
            ]
        }
    return classes_to_use[region]

def process_segmentation_file(data_path, dataset_path,label_to_use: list):
    filename = data_path.split(os.sep)[-2]
    if not filename.endswith('.mha'):
        filename = filename + '.mha'
    shutil.copy(data_path, os.path.join(dataset_path, f'labelsTr/{filename}'))

    # Load the segmentation file
    seg_image = sitk.ReadImage(os.path.join(dataset_path, f'labelsTr/{filename}'))
    seg_array = sitk.GetArrayFromImage(seg_image)
    new_seg_array = np.zeros_like(seg_array, dtype=seg_array.dtype)
    new_seg_array[seg_array != 0] = -1  # Initialize all non-background voxels to -1
    for i, label in enumerate(label_to_use):
        new_seg_array[seg_array == label] = i + 1
    # Filter the segmentation to keep only the classes of interest
    # All voxels not in label_to_use will be set to 0 (background)
    
    # Save the filtered segmentation back
    filtered_seg_image = sitk.GetImageFromArray(new_seg_array)
    filtered_seg_image.CopyInformation(seg_image)
    sitk.WriteImage(filtered_seg_image, os.path.join(dataset_path, f'labelsTr/{filename}'))

def process_file(data_path, dataset_path, modality_suffix="_0000", outsideValue=0):
    curr_img = sitk.ReadImage(data_path)
    filename = data_path.split(os.sep)[-2]
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

        

def create_dataset_json(config, preprocessing, dataset_data_path): 
    data_dataset_json = {
        "labels": {
            "label_001": "1", 
            "background": 0
        },
        "channel_names": {
            "0": preprocessing,
            # "1": config["preprocessing_mask"],
            
        },
        "numTraining": config['num_train'],
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

def move_gt_segmentations(dataset_target_path, nnunet_targets_preprocessed_dir):
    list_targets = glob.glob(os.path.join(f"{dataset_target_path}/imagesTr", '*'))
    list_targets.sort()
    list_gt_segmentations_datas = glob.glob(os.path.join(f"{nnunet_targets_preprocessed_dir}/gt_segmentations", '*'))
    list_gt_segmentations_datas.sort()

    print(nnunet_targets_preprocessed_dir)

    for (preprocessed_path, gt_path) in zip(list_targets, list_gt_segmentations_datas):
        # here, gt_path is the path to the gt_segmentation in nnUNet_preprocessed.
        print(preprocessed_path, "->", gt_path) # ensure correct file pairing; 
        shutil.copy(src = preprocessed_path, dst = gt_path) 

def move_masks(list_data_mask, dataset_mask_path):

    def _process_mask_file(data_path, dataset_mask_path):

        filename = data_path.split(os.sep)[-2]
        if not filename.endswith(f'.mha'):
            filename = filename + f'.mha'
        shutil.copy(data_path, os.path.join(dataset_mask_path, filename))


    # Use the affine from the last MRI as a placeholder, but for sitk we use spacing/origin/direction from the image itself
    os.makedirs(dataset_mask_path, exist_ok=True) 
    with ThreadPoolExecutor() as executor:
        list(tqdm(executor.map(lambda data_path: _process_mask_file(data_path, dataset_mask_path), list_data_mask), total=len(list_data_mask)))

def move_gt_target(list_data_ct, dataset_target_path2):

    def _process_target_file(data_path, dataset_target_path2):

        filename = data_path.split(os.sep)[-2]
        if not filename.endswith(f'.mha'):
            filename = filename + f'.mha'
        shutil.copy(data_path, os.path.join(dataset_target_path2, filename))


    # Use the affine from the last MRI as a placeholder, but for sitk we use spacing/origin/direction from the image itself
    os.makedirs(dataset_target_path2, exist_ok=True) 
    with ThreadPoolExecutor() as executor:
        list(tqdm(executor.map(lambda data_path: _process_target_file(data_path, dataset_target_path2), list_data_ct), total=len(list_data_ct)))

def modify_plan(plan_file_path, old_config_name, new_config_name, attribute_to_change, new_value):
    '''
    Making changes to preprocessing including patch sizes
    '''

    new_config =  {
            "inherits_from": old_config_name,
            "data_identifier": new_config_name,
            attribute_to_change: new_value
        }
    with open(plan_file_path) as f:
        plan = json.load(f)
    plan['configurations'][new_config_name] = new_config
    with open(plan_file_path, 'w') as f:
        json.dump(plan, f, indent=4)
    print(f'The plan {plan_file_path} has been updated. ')
