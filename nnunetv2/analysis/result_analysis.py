from glob import glob
import os
from tqdm import tqdm
import SimpleITK as sitk
import pandas as pd
import json
from image_metrics import ImageMetricsCompute
import matplotlib.pyplot as plt
from pathlib import Path
import shutil


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


class TestingResults():
    def __init__(self, pred_path, task=1, region='AB', save_path=None):
        if not save_path:
            save_path = pred_path+'_analysis'
        print(f'Save path: {save_path}')
        os.makedirs(save_path, exist_ok=True)

        self.pred_path = pred_path
        self.raw_image_path = f"/datasets/work/hb-synthrad2023/source/synthrad2025_data_v2/synthRAD2025_Task{task}_Train/Task{task}/{region}"
        self.save_path = save_path

        pred_files = sorted(glob(os.path.join(pred_path, '*.mha')))
        self.patient_ids = [Path(pred_file).stem for pred_file in pred_files]
        self.col_names = ['src', 'pred', 'gt', 'mask', 'error']

        # init image metrics
        self.test_metrics = ImageMetricsCompute()
        self.test_metrics.init_storage(["mae", "psnr", "ms_ssim"])

        # init save sub-folders
        self.slice_pc_to_save = [25, 50, 75]
        for pc in self.slice_pc_to_save:
            save_path_pc = os.path.join(self.save_path, '{}pc_png'.format(pc))
            if not os.path.exists(save_path_pc):
                os.makedirs(save_path_pc)
                print('Make path: {}'.format(save_path_pc))

        # all 3d images for analysis
        self.save_path_all_3d_img = os.path.join(self.save_path, 'all_3d_img')
        if not os.path.exists(self.save_path_all_3d_img):
            os.makedirs(self.save_path_all_3d_img)

    def process_patients(self):
        for patient_id in tqdm(self.patient_ids):
            self.process_a_patient(patient_id)
        self.analysis_patients()

    def analysis_patients(self):
        # save aggregated metrics
        dict_metric = self.test_metrics.aggregate()
        with open(os.path.join(self.save_path, 'results_overall_masked.json'), 'w') as f:
            json.dump(dict_metric, f, indent=4)

        # save individual metric
        df = pd.DataFrame(
            {
                'patient_id': self.test_metrics.storage_id,
                'mae': self.test_metrics.storage['mae'],
                'ms_ssim': self.test_metrics.storage['ms_ssim'],
                'psnr': self.test_metrics.storage['psnr'],
            }
        )
        df.to_csv(os.path.join(self.save_path, 'results_individual.csv'), index=True)
        print("mean mae:", dict_metric['mae']['mean'])
        print("mean psnr:", dict_metric['psnr']['mean'])
        print("mean ms_ssim:", dict_metric['ms_ssim']['mean'])
    
    def process_a_patient(self, patient_id):
        pred_path = os.path.join(self.pred_path, f'{patient_id}.mha')
        src_path = os.path.join(self.raw_image_path, patient_id, 'mr.mha')
        gt_path = os.path.join(self.raw_image_path, patient_id, 'ct.mha')
        mask_path = os.path.join(self.raw_image_path, patient_id,  'mask.mha')

        # read images
        img_src = sitk.ReadImage(src_path)
        img_pred = sitk.ReadImage(pred_path, sitk.sitkFloat32)
        img_gt = sitk.ReadImage(gt_path, sitk.sitkFloat32)
        img_mask = sitk.ReadImage(mask_path, sitk.sitkUInt8)

        # compute scores
        array_src = sitk.GetArrayFromImage(img_src)
        array_pred = sitk.GetArrayFromImage(img_pred)
        array_gt = sitk.GetArrayFromImage(img_gt)
        array_mask = sitk.GetArrayFromImage(img_mask)
        res = self.test_metrics.score_patient(array_gt, array_pred, array_mask)
        self.test_metrics.add(res, patient_id)

        # save error images
        self._save_error_image(img_pred, img_gt, img_mask, patient_id)
        self._copy_images(pred_path, src_path, gt_path, mask_path, patient_id)

        # save_png_slice
        self._save_png_slice(array_src, array_pred, array_gt, array_mask, patient_id, pc=50)
        self._save_png_slice(array_src, array_pred, array_gt, array_mask, patient_id, pc=25)
        self._save_png_slice(array_src, array_pred, array_gt, array_mask, patient_id, pc=75)
        plt.close('all')
    
    def _save_error_image(self, img_pred, img_gt, img_mask, patient_id):
        # save error images
        img_err = sitk.AbsoluteValueDifference(img_pred, img_gt)
        img_err = sitk.Mask(img_err, img_mask, outsideValue=0)
        img_err.CopyInformation(img_pred)
        sitk.WriteImage(img_err, os.path.join(self.save_path_all_3d_img, f'{patient_id}_error.mha'))
        # print('Save Error images: ', os.path.join(save_err_path, f'{patient_id}.mha'))
    
    def _copy_images(self, pred_path, src_path, gt_path, mask_path, patient_id):
        shutil.copy(pred_path, os.path.join(self.save_path_all_3d_img, f'{patient_id}_pred.mha'))
        shutil.copy(src_path, os.path.join(self.save_path_all_3d_img, f'{patient_id}_src.mha'))
        shutil.copy(gt_path, os.path.join(self.save_path_all_3d_img, f'{patient_id}_gt.mha'))
        shutil.copy(mask_path, os.path.join(self.save_path_all_3d_img, f'{patient_id}_mask.mha'))

    def _save_png_slice(self, array_src, array_pred, array_gt, array_mask, patient_id, pc=50):
        # init parameters
        slice_a0 = int(array_gt.shape[0] * pc / 100)
        slice_a1 = int(array_gt.shape[1] * pc / 100)
        slice_a2 = int(array_gt.shape[2] * pc / 100)
        rows = []

        row_slices = [slice_a0, slice_a1, slice_a2]
        # axial images
        slice_a0_src = array_src[slice_a0, :, :]
        slice_a0_pred = array_pred[slice_a0, :, :]
        slice_a0_gt = array_gt[slice_a0, :, :]
        slice_a0_mask = array_mask[slice_a0, :, :]
        slice_a0_error = slice_a0_gt-slice_a0_pred
        slice_a0_error[~slice_a0_mask.astype('bool')] = 0
        row_0 = [slice_a0_src, slice_a0_pred, slice_a0_gt, slice_a0_mask, slice_a0_error]
        rows.append(row_0)
        # coronal images
        slice_a1_src = array_src[:, slice_a1, :]
        slice_a1_pred = array_pred[:, slice_a1, :]
        slice_a1_gt = array_gt[:, slice_a1, :]
        slice_a1_mask = array_mask[:, slice_a1, :]
        slice_a1_error = slice_a1_gt - slice_a1_pred
        slice_a1_error[~slice_a1_mask.astype('bool')] = 0
        row_1 = [slice_a1_src, slice_a1_pred, slice_a1_gt, slice_a1_mask, slice_a1_error]
        rows.append(row_1)
        # sagital images
        slice_a2_src = array_src[:, :, slice_a2]
        slice_a2_pred = array_pred[:, :, slice_a2]
        slice_a2_gt = array_gt[:, :, slice_a2]
        slice_a2_mask = array_mask[:, :, slice_a2]
        slice_a2_error = slice_a2_gt - slice_a2_pred
        slice_a2_error[~slice_a2_mask.astype('bool')] = 0
        row_2 = [slice_a2_src, slice_a2_pred, slice_a2_gt, slice_a2_mask, slice_a2_error]
        rows.append(row_2)
        # plot
        fig, ax = plt.subplots(3, len(row_0), figsize=(15, 10))
        for row in range(3):
            for col in range(len(row_0)):
                if col < 4:
                    if col ==1 or col == 2:
                        ax[row, col].imshow(rows[row][col], cmap='gray', vmin=-1024, vmax=2000)
                    else:
                        ax[row, col].imshow(rows[row][col], cmap='gray')
                else:
                    ax[row, col].imshow(rows[row][col], cmap='twilight_shifted')
                ax[row, col].axis('off')
                ax[row, col].set_title('{}_slice{}'.format(self.col_names[col], row_slices[row]))
        fig.subplots_adjust(wspace=0.05, top=0.8)
        save_path = os.path.join(self.save_path, '{}pc_png' .format(pc))
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        fig.savefig(os.path.join(save_path, '{}.png'.format(patient_id)))
        # print('Save png slices: ', save_path)
        return fig

    





if __name__ == '__main__':

    # pred_path = "/datasets/work/hb-synthrad2023/work/synthrad2025/bw_workplace/data/nnunet_struct/results/Dataset206_synthrad2025_task1_MR_mednextL/nnUNetTrainerV2_MedNeXt_L_kernel3__nnUNetPlans__3d_fullres/fold_0/validation"
    # pred_path_revert_norm = pred_path + "_revert_norm"

    # raw_data_path = f"/datasets/work/hb-synthrad2023/work/synthrad2025/bw_workplace/data/nnunet_struct/preprocessed/Dataset206_synthrad2025_task1_MR_mednextL"
    # gt_path = os.path.join(raw_data_path, "gt_segmentations")
    # mask_path = os.path.join(raw_data_path, "masks")
    # results, df = compute_folder_metrics(pred_path_revert_norm, gt_path, mask_path)
    # print("mean mae:", results['mae']['mean'])
    # print("mean psnr:", results['psnr']['mean'])
    # print("mean ms_ssim:", results['ms_ssim']['mean'])

    input_path = "/datasets/work/hb-synthrad2023/source/synthrad2025_data_v2/synthRAD2025_Task1_Train/Task1/AB" # contain p_id/ct.mha, mask.mha, mr.mha
    pred_path_revert_norm = "/datasets/work/hb-synthrad2023/work/synthrad2025/bw_workplace/data/nnunet_struct/results/Dataset340_synthrad2025_task1_MR_AB_mednext/nnUNetTrainerV2_MedNeXt_L_kernel5__nnUNetPlans__3d_fullres/fold_0/validation_revert_norm"
    vs = TestingResults(pred_path_revert_norm, task=1, region='AB')
    vs.process_a_patient('1ABA011')
    # vs.process_patients()


