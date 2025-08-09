import torch
from tqdm.auto import tqdm
import os
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager, ConfigurationManager
from batchgenerators.utilities.file_and_folder_operations import load_json,join
from nnunetv2.utilities.label_handling.label_handling import determine_num_input_channels
from nnunetv2.utilities.find_class_by_name import recursive_find_python_class
from acvl_utils.cropping_and_padding.padding import pad_nd_image
from nnunetv2.inference.sliding_window_prediction import compute_gaussian, compute_steps_for_sliding_window
from totalsegmentator.alignment import undo_canonical
from totalsegmentator.resampling import change_spacing

from totalsegmentator.postprocessing import remove_auxiliary_labels
from nnunetv2.utilities.helpers import empty_cache
import nnunetv2
from pathlib import Path
import nibabel as nib
import time
import SimpleITK
# from resampling import change_spacing
import tempfile
import numpy as np
from nnunetv2.inference.export_prediction import convert_predicted_logits_to_segmentation_with_correct_shape
from nnunetv2.training.nnUNetTrainer.nnUNetTSTrainer import nnUNetTSTrainer





class MinialTotalSegmentator():
    def __init__(self, verbose=False):
        super().__init__()

        # path upated. by bx
        os.environ['nnUNet_raw'] = '/datasets/work/hb-synthrad2023/work/synthrad2025/bw_workplace/ref/evaluation/.totalsegmentator/nnunet/results'
        os.environ['nnUNet_preprocessed'] = '/datasets/work/hb-synthrad2023/work/synthrad2025/bw_workplace/ref/evaluation/.totalsegmentator/nnunet/results'
        os.environ['nnUNet_results'] = '/datasets/work/hb-synthrad2023/work/synthrad2025/bw_workplace/ref/evaluation/.totalsegmentator/nnunet/results'
        os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
        os.environ['KMP_INIT_AT_FORK'] = 'FALSE'

        self.verbose = verbose
        self.verbose_preprocessing = verbose
        self.allow_tqdm = verbose

        self.plans_manager, self.configuration_manager, self.list_of_parameters, self.network, self.dataset_json, \
        self.trainer_name, self.allowed_mirroring_axes, self.label_manager = None, None, None, None, None, None, None, None

        self.tile_step_size = 0.5
        self.use_gaussian = True
        self.use_mirroring = False
        device = torch.device('cpu')
        if device.type == 'cuda':
            torch.backends.cudnn.benchmark = True
        else:
            print(f'perform_everything_on_device=True is only supported for cuda devices! Setting this to False')
            perform_everything_on_device = False
        self.device = device
        self.perform_everything_on_device = perform_everything_on_device

        # path updated. by bx
        model_training_output_dir = '/datasets/work/hb-synthrad2023/work/synthrad2025/bw_workplace/ref/evaluation/.totalsegmentator/nnunet/results/Dataset297_TotalSegmentator_total_3mm_1559subj/nnUNetTrainer_4000epochs_NoMirroring__nnUNetPlans__3d_fullres'
        checkpoint_name = 'checkpoint_final.pth'
        dataset_json = load_json(join(model_training_output_dir, 'dataset.json'))
        plans = load_json(join(model_training_output_dir, 'plans.json'))
        plans_manager = PlansManager(plans)

        with torch.serialization.safe_globals([np.core.multiarray.scalar, np.dtype, np.dtypes.Float64DType,np.dtypes.Float32DType]):
            checkpoint = torch.load(join(model_training_output_dir, f'fold_0', checkpoint_name),
                            map_location=torch.device('cpu'), weights_only=False)
        configuration_name = checkpoint['init_args']['configuration']
        trainer_name = checkpoint['trainer_name']
        configuration_manager = plans_manager.get_configuration(configuration_name)

        num_input_channels = determine_num_input_channels(plans_manager, configuration_manager, dataset_json)
        trainer_class = recursive_find_python_class(join(nnunetv2.__path__[0], "training", "nnUNetTrainer"),
                                                    trainer_name, 'nnunetv2.training.nnUNetTrainer')
        inference_allowed_mirroring_axes = checkpoint['inference_allowed_mirroring_axes'] if \
            'inference_allowed_mirroring_axes' in checkpoint.keys() else None

        self.network = trainer_class.build_network_architecture(
            configuration_manager.network_arch_class_name,
            configuration_manager.network_arch_init_kwargs,
            configuration_manager.network_arch_init_kwargs_req_import,
            num_input_channels,
            plans_manager.get_label_manager(dataset_json).num_segmentation_heads,
            enable_deep_supervision=False
        )
        self.plans_manager = plans_manager
        self.configuration_manager = configuration_manager
        self.dataset_json = dataset_json
        self.trainer_name = trainer_name
        self.allowed_mirroring_axes = inference_allowed_mirroring_axes
        self.label_manager = plans_manager.get_label_manager(dataset_json)


        self.network.load_state_dict(checkpoint['network_weights'])
        self.network.eval()
        # self.network = torch.compile(self.network, backend="openvino")
        self.network.share_memory()
    def _internal_get_sliding_window_slicers(self, image_size):
        slicers = []
        if len(self.configuration_manager.patch_size) < len(image_size):
            assert len(self.configuration_manager.patch_size) == len(
                image_size) - 1, 'if tile_size has less entries than image_size, ' \
                                'len(tile_size) ' \
                                'must be one shorter than len(image_size) ' \
                                '(only dimension ' \
                                'discrepancy of 1 allowed).'
            steps = compute_steps_for_sliding_window(image_size[1:], self.configuration_manager.patch_size,
                                                    self.tile_step_size)
            for d in range(image_size[0]):
                for sx in steps[0]:
                    for sy in steps[1]:
                        slicers.append(
                            tuple([slice(None), d, *[slice(si, si + ti) for si, ti in
                                                    zip((sx, sy), self.configuration_manager.patch_size)]]))
        else:
            steps = compute_steps_for_sliding_window(image_size, self.configuration_manager.patch_size,
                                                     self.tile_step_size)
            for sx in steps[0]:
                for sy in steps[1]:
                    for sz in steps[2]:
                        slicers.append(
                            tuple([slice(None), *[slice(si, si + ti) for si, ti in
                                                zip((sx, sy, sz), self.configuration_manager.patch_size)]]))
        return slicers

    def _internal_maybe_mirror_and_predict(self, x: torch.Tensor) -> torch.Tensor:
        mirror_axes = self.allowed_mirroring_axes if self.use_mirroring else None
        prediction = self.network(x)

        if mirror_axes is not None:
            # check for invalid numbers in mirror_axes
            # x should be 5d for 3d images and 4d for 2d. so the max value of mirror_axes cannot exceed len(x.shape) - 3
            assert max(mirror_axes) <= x.ndim - 3, 'mirror_axes does not match the dimension of the input!'

            mirror_axes = [m + 2 for m in mirror_axes]
            axes_combinations = [
                c for i in range(len(mirror_axes)) for c in itertools.combinations(mirror_axes, i + 1)
            ]
            for axes in axes_combinations:
                prediction += torch.flip(self.network(torch.flip(x, axes)), axes)
            prediction /= (len(axes_combinations) + 1)
        return prediction

    def _internal_predict_sliding_window_return_logits(self,
                                                       data: torch.Tensor,
                                                       slicers,
                                                       do_on_device: bool = True,
                                                       ):
        predicted_logits = n_predictions = prediction = gaussian = workon = None
        results_device = self.device if do_on_device else torch.device('cpu')

        try:
            empty_cache(self.device)

            # move data to device
            if self.verbose:
                print(f'move image to device {results_device}')
            data = data.to(results_device) if torch.is_tensor(data) else torch.from_numpy(data).to(results_device)

            # preallocate arrays
            if self.verbose:
                print(f'preallocating results arrays on device {results_device}')
            predicted_logits = torch.zeros((self.label_manager.num_segmentation_heads, *data.shape[1:]),
                                           dtype=torch.half,
                                           device=results_device)
            n_predictions = torch.zeros(data.shape[1:], dtype=torch.half, device=results_device)

            if self.use_gaussian:
                gaussian = compute_gaussian(tuple(self.configuration_manager.patch_size), sigma_scale=1. / 8,
                                            value_scaling_factor=10,
                                            device=results_device)
            else:
                gaussian = 1

            if not self.allow_tqdm and self.verbose:
                print(f'running prediction: {len(slicers)} steps')
            for sl in tqdm(slicers, disable=not self.allow_tqdm):
                workon = data[sl][None]
                workon = workon.to(self.device)

                prediction = self._internal_maybe_mirror_and_predict(workon)[0].to(results_device)

                if self.use_gaussian:
                    prediction *= gaussian
                predicted_logits[sl] += prediction
                n_predictions[sl[1:]] += gaussian

            predicted_logits /= n_predictions
            # check for infs
            if torch.any(torch.isinf(predicted_logits)):
                raise RuntimeError('Encountered inf in predicted array. Aborting... If this problem persists, '
                                   'reduce value_scaling_factor in compute_gaussian or increase the dtype of '
                                   'predicted_logits to fp32')
        except Exception as e:
            del predicted_logits, n_predictions, prediction, gaussian, workon
            empty_cache(self.device)
            empty_cache(results_device)
            raise e
        return predicted_logits

    def score_patient(self, file_in, orientation, mask, resample=3.0, nr_threads_resampling=1, save_pred_seg_path=None):
        fname_id = Path(str(file_in)).stem #str(file_in).split('/')[-1].split('.')[0]
        with tempfile.TemporaryDirectory(prefix=f"nnunet_tmp_{fname_id}") as tmp_folder:
            tmp_dir = Path(tmp_folder)
            (tmp_dir / "mha").mkdir()
            # mha to nifti
            read = SimpleITK.ReadImage(file_in)
            if orientation is not None:
                spacing, origin, direction = orientation
                read.SetSpacing(spacing)
                read.SetOrigin(origin)
                read.SetDirection(direction)
            SimpleITK.WriteImage(read, str(tmp_dir / "mha" / "converted_mha.nii.gz"))

            # mha_to_nifti(file_in, tmp_dir / "mha" / "converted_mha.nii.gz", tmp_dir, orientation=orientation, verbose=verbose)
            file_in_mha = file_in
            file_in = tmp_dir / "mha" / "converted_mha.nii.gz"

            img_in_orig = nib.load(file_in)



            img_data = img_in_orig.get_fdata()
            img_data = img_data if mask is None else np.where(mask == 0, -1024, img_data)

            img_in = nib.Nifti1Image(img_data, img_in_orig.affine)  # copy img_in_orig
            img_in = nib.as_closest_canonical(img_in)
            if resample is not None:
                st = time.time()
                img_in_shape = img_in.shape
                img_in_zooms = img_in.header.get_zooms()
                img_in_rsp = change_spacing(img_in, resample,
                                            order=3, dtype=np.int32, nr_cpus=nr_threads_resampling)  # 4 cpus instead of 1 makes it a bit slower
                if self.verbose:
                    print(f"  from shape {img_in.shape} to shape {img_in_rsp.shape}")
                # if not quiet: print(f"  Resampled in {time.time() - st:.2f}s")
            else:
                img_in_rsp = img_in

            label_manager = self.plans_manager.get_label_manager(self.dataset_json)
            preprocessor = self.configuration_manager.preprocessor_class(verbose=self.verbose)
            data_properties = {
                'nibabel_stuff': {
                    'original_affine': img_in_rsp.affine,
                    'reoriented_affine': img_in_rsp.affine
                },
                'spacing': [3.0, 3.0, 3.0],
            }

            data, seg = preprocessor.run_case_npy(data=img_in_rsp.get_fdata().T[np.newaxis,...], seg=None, properties=data_properties, 
                                                               plans_manager=self.plans_manager,
                                                               configuration_manager=self.configuration_manager,
                                                               dataset_json=self.dataset_json)

            data = torch.from_numpy(data).to(dtype=torch.float32, memory_format=torch.contiguous_format)

            data, slicer_revert_padding = pad_nd_image(image=data, new_shape=self.configuration_manager.patch_size,
                                                           mode='constant', return_slicer=True,
                                                           shape_must_be_divisible_by=None)

            slicers = self._internal_get_sliding_window_slicers(data.shape[1:])
            predicted_logits = self._internal_predict_sliding_window_return_logits(data, slicers, False)
            empty_cache(self.device)
            # revert padding
            predicted_logits = predicted_logits[(slice(None), *slicer_revert_padding[1:])]
            img_pred = convert_predicted_logits_to_segmentation_with_correct_shape(
                    predicted_logits, self.plans_manager, self.configuration_manager, self.label_manager,data_properties,False)

            img_pred = nib.Nifti1Image(img_pred.T, img_in_rsp.affine)
            empty_cache(self.device)

            if resample is not None:
                if self.verbose: print(f"  back from {img_pred.shape} to original shape: {img_in_shape}")
                # Use force_affine otherwise output affine sometimes slightly off (which then is even increased
                # by undo_canonical)
                img_pred = change_spacing(img_pred, resample, img_in_shape,
                                          order=0, dtype=np.uint8, nr_cpus=nr_threads_resampling,
                                          force_affine=img_in.affine)
            if self.verbose: print("Undoing canonical...")
            empty_cache(self.device)
            img_pred = undo_canonical(img_pred, img_in_orig)
            empty_cache(self.device)

            img_data = img_pred.get_fdata().astype(np.uint8)
            if save_pred_seg_path is not None:
                if self.verbose: print(f"Saving predicted segmentation to {save_pred_seg_path}")
                nib.save(img_pred, os.path.join(save_pred_seg_path, f"{fname_id}.nii.gz"))
            return img_data
        