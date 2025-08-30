from glob import glob
import os
from tqdm import tqdm
import SimpleITK as sitk
import pandas as pd
import json
import matplotlib.pyplot as plt
from pathlib import Path
import shutil
from nnunetv2.analysis.image_metrics import ImageMetricsCompute
import numpy as np    





class ValidationResults():
    """
    Class to analyze the results of the predictions.
    It computes the metrics and saves the results in a folder.
    This is used directly in the nnUNetTrainerMRCT class.
    """

    def __init__(self, pred_path, gt_path, mask_path, src_path=None, gt_segmentation_path=None, save_path=None, save_pred_seg_path=None):
        if not save_path:
            save_path = pred_path+'_analysis'
        if gt_segmentation_path:
            save_pred_seg_path = os.path.join(save_path, 'predicted_segmentations')
            if not os.path.exists(save_pred_seg_path):
                os.makedirs(save_pred_seg_path)
                

            print(f'Saving predicted segmentations to: {save_pred_seg_path}')
        self.post_fix = 'nii.gz'
        print(f'Save path: {save_path}')
        os.makedirs(save_path, exist_ok=True)
        self.save_path = save_path
        self.save_pred_seg_path = save_pred_seg_path

        self.pred_path = pred_path
        self.gt_path = gt_path
        self.mask_path = mask_path
        self.src_path = src_path
        self.gt_segmentation_path = gt_segmentation_path
        # self.gt_segmentation_path = None # TODO REMOVE LATER

        pred_files = sorted(glob(os.path.join(pred_path, f'*.{self.post_fix}')))
        self.patient_ids = [Path(pred_file).stem for pred_file in pred_files]

        # init image metrics
        self.image_metrics = ImageMetricsCompute()
        self.image_metrics.init_storage(["mae", "psnr", "ms_ssim"])

        if self.gt_segmentation_path:
            print(f'Using segmentation metrics from: {self.gt_segmentation_path}')
            # init segmentation metrics
            from nnunetv2.analysis.segmentation_metrics import SegmentationMetricsCompute    

            self.seg_metrics = SegmentationMetricsCompute()
            self.seg_metrics.init_storage(["DICE", "HD95"])
    
    def process_patients_mp(self, max_workers=8):
        """
        Process patients in parallel using ThreadPoolExecutor.
        This method is used to speed up the processing of multiple patients.
        """
        from concurrent.futures import ThreadPoolExecutor
        from tqdm import tqdm

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            list(tqdm(executor.map(self.process_a_patient, self.patient_ids), total=len(self.patient_ids)))
        
        dict_metric = self.analysis_patients()
        return dict_metric
    
    def process_patients(self):
        for patient_id in tqdm(self.patient_ids):
            self.process_a_patient(patient_id)
        dict_metric = self.analysis_patients()
        return dict_metric

    def analysis_patients(self):
        # save aggregated metrics
        dict_metric = self.image_metrics.aggregate()
        if self.gt_segmentation_path:
            dict_metric_seg = self.seg_metrics.aggregate()
            dict_metric.update(dict_metric_seg)
        with open(os.path.join(self.save_path, 'results_overall_masked.json'), 'w') as f:
            json.dump(dict_metric, f, indent=4)

        # save individual metric
        df = pd.DataFrame(
            {
                'patient_id': self.image_metrics.storage_id,
                'mae': self.image_metrics.storage['mae'],
                'ms_ssim': self.image_metrics.storage['ms_ssim'],
                'psnr': self.image_metrics.storage['psnr'],
            }
        )
        if self.gt_segmentation_path:
            df['DICE'] = self.seg_metrics.storage['DICE']
            df['HD95'] = self.seg_metrics.storage['HD95']
        df.to_csv(os.path.join(self.save_path, 'results_individual.csv'), index=True)
        
        # print results
        print("mean mae:", dict_metric['mae']['mean'])
        print("mean psnr:", dict_metric['psnr']['mean'])
        print("mean ms_ssim:", dict_metric['ms_ssim']['mean'])
        if self.gt_segmentation_path:
            print("mean DICE:", dict_metric['DICE']['mean'])
            print("mean HD95:", dict_metric['HD95']['mean'])
        return dict_metric

    def process_a_patient(self, patient_id):
        pred_path = os.path.join(self.pred_path, f'{patient_id}.{self.post_fix}')
        gt_path = os.path.join(self.gt_path, f'{patient_id}.{self.post_fix}')
        mask_path = os.path.join(self.mask_path, f'{patient_id}.{self.post_fix}')

        # read images
        img_pred = sitk.ReadImage(pred_path, sitk.sitkFloat32)
        img_gt = sitk.ReadImage(gt_path, sitk.sitkFloat32)
        img_mask = sitk.ReadImage(mask_path, sitk.sitkUInt8)

        # compute image scores
        array_pred = sitk.GetArrayFromImage(img_pred)
        array_gt = sitk.GetArrayFromImage(img_gt)
        array_mask = sitk.GetArrayFromImage(img_mask)
        res = self.image_metrics.score_patient(array_gt, array_pred, array_mask)
        self.image_metrics.add(res, patient_id)

        # compute segmentation scores
        if self.gt_segmentation_path:
            mask_transposed = load_image_file_directly(location=mask_path)
            gt_segmentation_path = os.path.join(self.gt_segmentation_path, f'{patient_id}.{self.post_fix}')
            gt_segmentation_transposed = load_image_file_directly(location=gt_segmentation_path)
            try: 
                res_seg = self.seg_metrics.score_patient_ts(pred_path, mask_transposed, gt_segmentation_transposed, patient_id, save_pred_seg_path = self.save_pred_seg_path)
                self.seg_metrics.add(res_seg, patient_id)

            except Exception as e:
                print(f"!!!Error processing patient {patient_id}: {e}")
                print(f'!!!No label found for patient {patient_id}, skipping...')
                res_seg = {'DICE': np.nan, 'HD95': np.nan}
                self.seg_metrics.add(res_seg, patient_id)

    def aim_log_one_patient(self, aim_run, epoch, max_images=2):
        """
        Log the metrics of one patient to Aim.
        This is used to log the metrics of each patient during training.
        """
        def _float2uint8(array):
            """
            Convert a float array to uint8.
            This is used to convert the image arrays to uint8 for Aim logging.
            """
            array_norm = (array - array.min()) / (array.max() - array.min())
            return (array_norm * 255).astype('uint8')
        
        import aim
        train_images_list = []
        train_targets_list = []
        train_output_list = []
        for i, patient_id in enumerate(self.patient_ids[:max_images]):
            src_path = os.path.join(self.src_path, f'{patient_id}_0000.{self.post_fix}')
            pred_path = os.path.join(self.pred_path, f'{patient_id}.{self.post_fix}')
            gt_path = os.path.join(self.gt_path, f'{patient_id}.{self.post_fix}') 

            # read images
            img_pred = sitk.ReadImage(pred_path, sitk.sitkFloat32)
            img_gt = sitk.ReadImage(gt_path, sitk.sitkFloat32)
            img_src = sitk.ReadImage(src_path, sitk.sitkFloat32)
            array_pred = sitk.GetArrayFromImage(img_pred)
            array_gt = sitk.GetArrayFromImage(img_gt)
            array_src = sitk.GetArrayFromImage(img_src)

            slice_to_save = int(array_gt.shape[0] * 50 / 100)
            train_images_list.append(
                aim.Image(_float2uint8(array_src[slice_to_save, :, :]), caption=f"Input Image: {i}"))
            train_targets_list.append(
                aim.Image(_float2uint8(array_gt[slice_to_save, :, :]), caption=f"Target Image: {i}"))
            train_output_list.append(
                aim.Image(_float2uint8(array_pred[slice_to_save, :, :]), caption=f"Predicted Label: {i}"))

        # tracking input, label and output images with Aim
        aim_run.track(
            train_images_list,
            name="validation",
            context={"type": "input"},
            step=epoch 
        )
        aim_run.track(
            train_targets_list,
            name="validation",
            context={"type": "target"},
            step=epoch 
        )
        aim_run.track(
            train_output_list,
            name="validation",
            context={"type": "prediction"},
            step=epoch
        )
        


class FinalValidationResults(ValidationResults):
    """
    Class to analyze the results of the final validation predictions.
    It computes the metrics and saves the results in a folder.
    This is used directly in the nnUNetTrainerMRCT class.
    """
    
    def __init__(self, pred_path, gt_path, mask_path, src_path=None, gt_segmentation_path=None, save_path=None, save_pred_seg_path=None):
        super().__init__(pred_path, gt_path, mask_path, src_path, gt_segmentation_path, save_path, save_pred_seg_path)
        self.src_path = src_path
        self.save_path_all_3d_img = os.path.join(self.save_path, 'all_3d_img')
        if not os.path.exists(self.save_path_all_3d_img):
            os.makedirs(self.save_path_all_3d_img)

        # for saving images
        self.col_names = ['src', 'pred', 'gt', 'mask', 'error']
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

    def process_a_patient(self, patient_id):
        pred_path = os.path.join(self.pred_path, f'{patient_id}.{self.post_fix}')
        gt_path = os.path.join(self.gt_path, f'{patient_id}.{self.post_fix}') 
        mask_path = os.path.join(self.mask_path, f'{patient_id}.{self.post_fix}')
        src_path = os.path.join(self.src_path, f'{patient_id}_0000.{self.post_fix}')

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
        res = self.image_metrics.score_patient(array_gt, array_pred, array_mask)
        self.image_metrics.add(res, patient_id)

        # compute segmentation scores
        if self.gt_segmentation_path:
            mask_transposed = load_image_file_directly(location=mask_path)
            gt_segmentation_path = os.path.join(self.gt_segmentation_path, f'{patient_id}.{self.post_fix}')
            gt_segmentation_transposed = load_image_file_directly(location=gt_segmentation_path)


            try: 
                res_seg = self.seg_metrics.score_patient_ts(pred_path, mask_transposed, gt_segmentation_transposed, patient_id, save_pred_seg_path = self.save_pred_seg_path)
                self.seg_metrics.add(res_seg, patient_id)

            except Exception as e:
                print(f"!!!Error processing patient {patient_id}: {e}")
                print(f'!!!No label found for patient {patient_id}, skipping...')
                res_seg = {'DICE': np.nan, 'HD95': np.nan}
                self.seg_metrics.add(res_seg, patient_id)


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
        sitk.WriteImage(img_err, os.path.join(self.save_path_all_3d_img, f'{patient_id}_error.{self.post_fix}'))
        # print('Save Error images: ', os.path.join(save_err_path, f'{patient_id}.mha'))
    
    def _copy_images(self, pred_path, src_path, gt_path, mask_path, patient_id):
        shutil.copy(pred_path, os.path.join(self.save_path_all_3d_img, f'{patient_id}_pred.{self.post_fix}'))
        shutil.copy(src_path, os.path.join(self.save_path_all_3d_img, f'{patient_id}_src.{self.post_fix}'))
        shutil.copy(gt_path, os.path.join(self.save_path_all_3d_img, f'{patient_id}_gt.{self.post_fix}'))
        shutil.copy(mask_path, os.path.join(self.save_path_all_3d_img, f'{patient_id}_mask.{self.post_fix}'))
        if self.gt_segmentation_path and self.save_pred_seg_path:
            gt_segmentation_path = os.path.join(self.gt_segmentation_path, f'{patient_id}.{self.post_fix}')
            shutil.copy(gt_segmentation_path, os.path.join(self.save_pred_seg_path, f'{patient_id}_gt_segmentation.{self.post_fix}'))

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



def load_image_file_directly(*, location, return_orientation=False, set_orientation=None):
    # immediatly load the file and find its orientation
    result = sitk.ReadImage(location)
    # Note, transpose needed because Numpy is ZYX according to SimpleITKs XYZ
    img_arr = np.transpose(sitk.GetArrayFromImage(result), [2, 1, 0])

    if return_orientation:
        spacing = result.GetSpacing()
        origin = result.GetOrigin()
        direction = result.GetDirection()


        return img_arr, spacing, origin, direction
    else:
        # If desired, force the orientation on an image before converting to NumPy array
        if set_orientation is not None:
            spacing, origin, direction = set_orientation
            result.SetSpacing(spacing)
            result.SetOrigin(origin)
            result.SetDirection(direction)

        # Note, transpose needed because Numpy is ZYX according to SimpleITKs XYZ
        return np.transpose(sitk.GetArrayFromImage(result), [2, 1, 0])
    





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

    nnUNet_root = '/datasets/work/hb-nhmrc-dhcp/work/xin015/data/dHCP_origin/nnunet_struct'
    nnUNet_preprocessed = f"{nnUNet_root}/preprocessed"
    nnUNet_raw = f"{nnUNet_root}/raw"
    nnUNet_results = f"{nnUNet_root}/results"

    dataset_name = "Dataset200_svr_calibration"
    pred_path_revert_norm = os.path.join(nnUNet_results, dataset_name, "nnUNetTrainerMRCT__nnUNetPlans__3d_fullres/fold_0/validation")

    gt_path = os.path.join(nnUNet_preprocessed, dataset_name, "gt_target")
    mask_path = os.path.join(nnUNet_preprocessed, dataset_name, "masks")
    # gt_segmentation_path = os.path.join(nnUNet_preprocessed, dataset_name, "gt_target_segmentation_ts")
    # gt_segmentation_path = None
    src_path = os.path.join(nnUNet_raw, dataset_name, 'imagesTr')

    ts = ValidationResults(pred_path_revert_norm, gt_path, mask_path, src_path, gt_segmentation_path=None)
    # ts.process_a_patient('2ABA044')
    ts.process_patients_mp()


