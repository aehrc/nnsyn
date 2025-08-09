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



def process_ct_file(data_path, dataset_path, modality_suffix="_0000"):
    filename = data_path.split(os.sep)[-2]
    if not filename.endswith(f'{modality_suffix}.mha'):
        filename = filename + f'{modality_suffix}.mha'
    shutil.copy(data_path, os.path.join(dataset_path, f'imagesTr/{filename}'))


def process_segmentation_file(data_path, dataset_path,label_to_use: list):
    filename = data_path.split(os.sep)[-2]
    if not filename.endswith('.mha'):
        filename = filename + '.mha'
    shutil.copy(data_path, os.path.join(dataset_path, f'labelsTr/{filename}'))

    # Load the segmentation file
    seg_image = sitk.ReadImage(os.path.join(dataset_path, f'labelsTr/{filename}'))
    seg_array = sitk.GetArrayFromImage(seg_image)
    new_seg_array = np.zeros_like(seg_array, dtype=np.uint8)
    for i, label in enumerate(label_to_use):
        new_seg_array[seg_array == label] = i + 1

    # Filter the segmentation to keep only the classes of interest
    # All voxels not in label_to_use will be set to 0 (background)
    
    # Save the filtered segmentation back
    filtered_seg_image = sitk.GetImageFromArray(new_seg_array)
    filtered_seg_image.CopyInformation(seg_image)
    sitk.WriteImage(filtered_seg_image, os.path.join(dataset_path, f'labelsTr/{filename}'))

        

def create_dataset_json(num_train, preprocessing, dataset_data_path, label_to_use): 
    labels = {str(label): i + 1 for i, label in enumerate(label_to_use)}
    labels["background"] = 0

    data_dataset_json = {
        "labels": labels,
        "channel_names": {
            "0": preprocessing,
            
        },
        "numTraining": num_train,
        "file_ending": ".mha"
    }
    dump_data_datasets_path = os.path.join(dataset_data_path, 'dataset.json')
    with open(dump_data_datasets_path, 'w') as f:
        json.dump(data_dataset_json, f)

def copy_dataset_json(raw_path, preprocessed_path, dataset_name):
    src_path = os.path.join(raw_path, dataset_name, 'dataset.json')
    dst_path = os.path.join(preprocessed_path, dataset_name, 'dataset.json')
    if not os.path.exists(dst_path):
        shutil.copy(src_path, dst_path)
    else:
        print(f"Dataset JSON already exists at {dst_path}, skipping copy.")


def check_plan_match(plan_syn, plan_seg, configuration='3d_fullres', item='patch_size'):
    """
    Check if the patch sizes in the two plans match.
    """
    # load json plans
    def _load_json(plan_path):
        with open(plan_path, 'r') as f:
            return json.load(f)
    plan_syn = _load_json(plan_syn)
    plan_seg = _load_json(plan_seg)

    if plan_syn["configurations"][configuration][item] != plan_seg["configurations"][configuration][item]:
        raise ValueError(f"Patch size mismatch: {plan_syn['configurations'][configuration][item]} vs {plan_seg['configurations'][configuration][item]}")
    else:
        print(f"Patch size match: {plan_syn['configurations'][configuration][item]} == {plan_seg['configurations'][configuration][item]}")
