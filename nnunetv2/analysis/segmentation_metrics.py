#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
from typing import Optional
import nibabel as nib
import os
import torch
import SimpleITK as sitk
from monai.metrics import DiceMetric, HausdorffDistanceMetric
from nibabel.nifti1 import Nifti1Image
from nnunetv2.analysis.ts_utils import MinialTotalSegmentator



class SegmentationMetrics():
    def __init__(self, debug=False):
        # Use fixed wide dynamic range
        self.debug = debug
        self.dynamic_range = [-1024., 3000.]
        self.my_ts = MinialTotalSegmentator(verbose=self.debug)

        # TotalSegmentator classes. See here https://github.com/wasserth/TotalSegmentator?tab=readme-ov-file#class-details (TotalSegmenator commit cd3d5362245237f13adbb78cdfaee615f54096a1)
        self.classes_to_use = {
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

    
    def score_patient_ts(self, synthetic_ct_location, mask, gt_segmentation, patient_id, orientation=None, save_pred_seg_path=None):
        with torch.no_grad():
            pred_seg=self.my_ts.score_patient(synthetic_ct_location, orientation, mask, save_pred_seg_path=save_pred_seg_path)
        # Retrieve the data in the NiftiImage from nibabel
        if isinstance(pred_seg, Nifti1Image):
            pred_seg = np.array(pred_seg.get_fdata())

        return self.score_patient(gt_segmentation, pred_seg, mask, patient_id, orientation)


    
    def score_patient(self, gt_segmentation, sct_segmentation, mask, patient_id, orientation=None):        
        # Calculate segmentation metrics
        # Perform segmentation using TotalSegmentator, enforce the orientation of the ground-truth on the output

        anatomy = patient_id[1:3].upper()

        assert sct_segmentation.shape == gt_segmentation.shape

        # Convert to PyTorch tensors for MONAI
        gt_seg = gt_segmentation.cpu().detach() if torch.is_tensor(gt_segmentation) else torch.from_numpy(gt_segmentation).cpu().detach()
        pred_seg = sct_segmentation.cpu().detach() if torch.is_tensor(sct_segmentation) else torch.from_numpy(sct_segmentation).cpu().detach()


        assert gt_seg.shape == pred_seg.shape
        if orientation is not None:
            spacing, origin, direction = orientation
        else:
            spacing=None
        
        # list of metrics to evaluate
        metrics = [
            {
                'name': 'DICE',
                'f':DiceMetric(include_background=True, reduction="mean", get_not_nans=False)
            }, {
                'name': 'HD95',
                'f': HausdorffDistanceMetric(include_background=True, reduction="mean", percentile=95, get_not_nans=False),
                'kwargs': {'spacing': spacing}
            }
        ]

        # Evaluate each one-hot metric 
        for c in self.classes_to_use[anatomy]:
            gt_tensor = (gt_seg == c).view(1, 1, *gt_seg.shape)
            if gt_tensor.sum() == 0:
                if self.debug:
                    print(f"No {c} in {patient_id}")
                continue
            est_tensor = (pred_seg == c).view(1, 1, *pred_seg.shape)
            for metric in metrics:
                metric['f'](est_tensor, gt_tensor, **metric['kwargs'] if 'kwargs' in metric else {})

        # aggregate the mean metrics for the patient over the classes
        result = {}
        for metric in metrics:
            result[metric['name']] = metric['f'].aggregate().item()
            metric['f'].reset()
        return result
    
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


class SegmentationMetricsCompute(SegmentationMetrics):
    """
    This class is used to compute the segmentation metrics for a patient.
    It inherits from SegmentationMetrics and overrides the score_patient method.
    """
    def __init__(self, debug=False):
        super().__init__(debug=debug)
        self.names = ['DICE', 'HD95']

    def init_storage(self, names: list):
        self.storage = dict()
        self.storage_id = []
        self.names = names
        for name in names:
            self.storage[name] = []

    def add(self, res: dict, patient_id=None):
        for key, value in res.items():
            self.storage[key].append(value)
        if patient_id:
            self.storage_id.append(patient_id)

    def aggregate(self):
        res = dict()
        for name in self.names:
            res[name] = dict()

        for key, value in self.storage.items():
            res[key]['mean'] = np.mean(value)
            res[key]['std'] = np.std(value)
            res[key]['max'] = np.max(value)
            res[key]['min'] = np.min(value)
            res[key]['25pc'] = np.percentile(value, 25)
            res[key]['50pc'] = np.percentile(value, 50)
            res[key]['75pc'] = np.percentile(value, 75)
            res[key]['count'] = len(value)
        return res

    def reset(self):
        for key, value in self.storage.items():
            self.storage[key] = []



if __name__ == "__main__":
    # Example usage
    # metrics = SegmentationMetrics(debug=True)
    # gt_segmentation_path = "/datasets/work/hb-synthrad2023/work/synthrad2025/bw_workplace/data/nnunet_struct/raw/Dataset800_SEGMENTATION_synthrad2025_task1_CT_AB_aligned_to_Dataset261/labelsTr/1ABA005.mha"
    # sct_segmentation_path = "/datasets/work/hb-synthrad2023/work/synthrad2025/bw_workplace/data/nnunet_struct/raw/Dataset800_SEGMENTATION_synthrad2025_task1_CT_AB_aligned_to_Dataset261/labelsTr/1ABA005.mha"
    # gt_segmentation = sitk.GetArrayFromImage(sitk.ReadImage(gt_segmentation_path))
    # sct_segmentation = sitk.GetArrayFromImage(sitk.ReadImage(sct_segmentation_path))
    # mask = None  # Example mask (not used in this example)
    # patient_id = "1ABA005"  # Example patient ID
    # orientation = None  # Example orientation (not used in this example)

    # result = metrics.score_patient(gt_segmentation, sct_segmentation, mask, patient_id, orientation)
    # print(result)


    # # real example
    # _segmentation_evaluator = SegmentationMetrics(debug=True)

    # patient_id = "1ABA011"
    # gt_segmentation_path = f"/datasets/work/hb-synthrad2023/work/synthrad2025/bw_workplace/data/nnunet_struct/raw/Dataset251_synthrad2025_task1_CT_AB_pre_v2r_stitched_masked_synseg/labelsTr/{patient_id}.mha"
    # gt_segmentation, spacing, origin, direction = load_image_file_directly(location=gt_segmentation_path, return_orientation=True)

    # # synthetic_ct_location = f"/datasets/work/hb-synthrad2023/work/synthrad2025/bw_workplace/data/nnunet_struct/raw/Dataset251_synthrad2025_task1_CT_AB_pre_v2r_stitched_masked_synseg/imagesTr/{patient_id}_0000.mha"
    # synthetic_ct_location = "/datasets/work/hb-synthrad2023/work/synthrad2025/bw_workplace/data/nnunet_struct/results/Dataset280_synthrad2025_task1_MR_AB_pre_v2r_stitched/nnUNetTrainerMRCT_track__nnUNetPlans__3d_fullres/fold_0/validation_revert_norm/1ABA011.mha"

    # # mask = None
    # mask = load_image_file_directly(location=f"/datasets/work/hb-synthrad2023/work/synthrad2025/bw_workplace/data/nnunet_struct/preprocessed/Dataset260_synthrad2025_task1_MR_AB_pre_v2r_stitched_masked/masks/{patient_id}.mha", set_orientation=(spacing, origin, direction))

    # seg_metrics = _segmentation_evaluator.score_patient_ts(synthetic_ct_location, mask, gt_segmentation, patient_id, orientation=(spacing, origin, direction))
    # print(f"Segmentation metrics for patient {patient_id}: {seg_metrics}")
    # # if we are in test phase, there is a doseplan for every patient in this folder

    # real example without orientation
    _segmentation_evaluator = SegmentationMetrics(debug=True)

    patient_id = "1ABA011"
    gt_segmentation_path = f"/datasets/work/hb-synthrad2023/work/synthrad2025/bw_workplace/data/nnunet_struct/raw/Dataset251_synthrad2025_task1_CT_AB_pre_v2r_stitched_masked_synseg/labelsTr/{patient_id}.mha"
    gt_segmentation = load_image_file_directly(location=gt_segmentation_path)

    # synthetic_ct_location = f"/datasets/work/hb-synthrad2023/work/synthrad2025/bw_workplace/data/nnunet_struct/raw/Dataset251_synthrad2025_task1_CT_AB_pre_v2r_stitched_masked_synseg/imagesTr/{patient_id}_0000.mha"
    synthetic_ct_location = "/datasets/work/hb-synthrad2023/work/synthrad2025/bw_workplace/data/nnunet_struct/results/Dataset280_synthrad2025_task1_MR_AB_pre_v2r_stitched/nnUNetTrainerMRCT_track__nnUNetPlans__3d_fullres/fold_0/validation_revert_norm/1ABA011.mha"

    # mask = None
    mask = load_image_file_directly(location=f"/datasets/work/hb-synthrad2023/work/synthrad2025/bw_workplace/data/nnunet_struct/preprocessed/Dataset260_synthrad2025_task1_MR_AB_pre_v2r_stitched_masked/masks/{patient_id}.mha")

    seg_metrics = _segmentation_evaluator.score_patient_ts(synthetic_ct_location, mask, gt_segmentation, patient_id)
    print(f"Segmentation metrics for patient {patient_id}: {seg_metrics}")