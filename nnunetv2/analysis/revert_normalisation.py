import sys
import SimpleITK as sitk
import json
import glob
import os
from tqdm import tqdm
import numpy as np
import torch


# revert normalisation
def get_ct_normalisation_values(ct_plan_path):
    """
    Get the mean and standard deviation for CT normalisation.
    """
    # Load the nnUNet plans file for CT
    with open(ct_plan_path, "r") as f:
        ct_plan = json.load(f)

    ct_mean = ct_plan['foreground_intensity_properties_per_channel']["0"]['mean']
    ct_std = ct_plan['foreground_intensity_properties_per_channel']["0"]['std']
    print(f"CT mean: {ct_mean}, CT std: {ct_std}")
    return ct_mean, ct_std

def revert_normalisation(pred_path, ct_mean, ct_std, save_path=None):
    """
    Revert the normalisation of a CT image.
    """
    if save_path is None:
        save_path = pred_path + '_revert_norm'
    os.makedirs(save_path, exist_ok=True)
    imgs = glob.glob(os.path.join(pred_path, "*.mha"))
    for img in tqdm(imgs):
        img_sitk = sitk.ReadImage(img)
        img_array = sitk.GetArrayFromImage(img_sitk)
        img_array = img_array * ct_std + ct_mean
        img_sitk = sitk.GetImageFromArray(img_array)
        img_sitk.CopyInformation(img_sitk)
        sitk.WriteImage(img_sitk, os.path.join(save_path, os.path.basename(img)))
        # print(f"Reverted saved to {os.path.join(save_path, os.path.basename(img))}")

if __name__ == "__main__":
    ct_plan_path = "/datasets/work/hb-synthrad2023/work/synthrad2025/bw_workplace/data/nnunet_struct/preprocessed/Dataset203_synthrad2025_task1_CT/nnUNetPlans.json"
    ct_mean, ct_std = get_ct_normalisation_values(ct_plan_path)
    pred_path = "/datasets/work/hb-synthrad2023/work/synthrad2025/bw_workplace/data/nnunet_struct/results/Dataset202_synthrad2025_task1_MR_mask/nnUNetTrainerMRCT__nnUNetPlans__3d_fullres/fold_0/validation"
    revert_normalisation(pred_path, ct_mean, ct_std, save_path=pred_path + "_revert_norm")
