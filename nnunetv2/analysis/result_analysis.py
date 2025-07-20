from glob import glob
import os
from tqdm import tqdm
import SimpleITK as sitk
import pandas as pd
import json
from image_metrics import ImageMetricsCompute

def compute_folder_metrics(pred_path, gt_path, mask_path):
    """
    Analyze the results of the predictions.
    """
    pred_files = sorted(glob(os.path.join(pred_path, '*.mha')))
    # gt_path = os.path.join(raw_data_path, "gt_segmentations")
    # mask_path = os.path.join(raw_data_path, "masks")
    testing_metrics = ImageMetricsCompute()
    testing_metrics.init_storage(["mae", "psnr", "ms_ssim"])

    for pred_file in tqdm(pred_files):
        filename = os.path.basename(pred_file)
        gt_file = os.path.join(gt_path, filename)
        mask_file = os.path.join(mask_path, filename)

        img_pred = sitk.ReadImage(pred_file)
        img_gt = sitk.ReadImage(gt_file)
        img_mask = sitk.ReadImage(mask_file, sitk.sitkUInt8)

        array_pred = sitk.GetArrayFromImage(img_pred)
        array_gt = sitk.GetArrayFromImage(img_gt)
        array_mask = sitk.GetArrayFromImage(img_mask)

        res = testing_metrics.score_patient(array_gt, array_pred, array_mask)
        testing_metrics.add(res, filename)

    # aggregate results
    results = testing_metrics.aggregate()

    df = pd.DataFrame(
            {
                'patient_id': testing_metrics.storage_id,
                'mae': testing_metrics.storage['mae'],
                'ms_ssim': testing_metrics.storage['ms_ssim'],
                'psnr': testing_metrics.storage['psnr'],
            }
        )

    # save results, and df in the folder "results"
    results_path = os.path.join(pred_path, "results.json")
    df_path = os.path.join(pred_path, "results_individual.csv")
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=4)
    df.to_csv(df_path, index=False)
    return results, df


if __name__ == '__main__':

    pred_path = "/datasets/work/hb-synthrad2023/work/synthrad2025/bw_workplace/data/nnunet_struct/results/Dataset206_synthrad2025_task1_MR_mednextL/nnUNetTrainerV2_MedNeXt_L_kernel3__nnUNetPlans__3d_fullres/fold_0/validation"
    pred_path_revert_norm = pred_path + "_revert_norm"

    raw_data_path = f"/datasets/work/hb-synthrad2023/work/synthrad2025/bw_workplace/data/nnunet_struct/preprocessed/Dataset206_synthrad2025_task1_MR_mednextL"
    gt_path = os.path.join(raw_data_path, "gt_segmentations")
    mask_path = os.path.join(raw_data_path, "masks")
    results, df = compute_folder_metrics(pred_path_revert_norm, gt_path, mask_path)
    print("mean mae:", results['mae']['mean'])
    print("mean psnr:", results['psnr']['mean'])
    print("mean ms_ssim:", results['ms_ssim']['mean'])