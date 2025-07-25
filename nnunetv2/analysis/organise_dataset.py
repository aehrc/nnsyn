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

def process_file_masked(data_path, dataset_path, modality_suffix="_0000", outsideValue=0):
    curr_img = sitk.ReadImage(data_path)
    mask_img = sitk.ReadImage(data_path.replace('mr.mha', 'mask.mha'), sitk.sitkUInt8)
    mask_img = sitk.Cast(mask_img, sitk.sitkUInt8)
    masked_image = sitk.Mask(curr_img, mask_img, outsideValue=outsideValue)
    # values in the mask different from maskingValue are copied over, all other values are set to outsideValue
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

def move_preprocessed(nnunet_datas_preprocessed_dir, nnunet_targets_preprocessed_dir, config): 
    list_preprocessed_datas_seg_path = sorted(glob.glob(os.path.join(nnunet_targets_preprocessed_dir, f'{config["plan"]}_{config["configuration"]}/*_seg.npy')))
    list_preprocessed_targets_path = sorted(glob.glob(os.path.join(nnunet_datas_preprocessed_dir, f'{config["plan"]}_{config["configuration"]}/*.npy')))
    list_preprocessed_targets_path = [name for name in list_preprocessed_targets_path if '_seg' not in name]

    for (datas_path, targets_path) in zip(list_preprocessed_datas_seg_path, list_preprocessed_targets_path):
        print(targets_path, "->", datas_path)
        shutil.copy(src = targets_path, dst = datas_path) 