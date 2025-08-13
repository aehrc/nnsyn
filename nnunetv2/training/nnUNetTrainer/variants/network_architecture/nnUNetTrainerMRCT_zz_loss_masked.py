import torch
from typing import Union, Tuple, List


from nnunetv2.training.loss.mse import myMSE, myMaskedMSE

from nnunetv2.training.dataloading.data_loader_2d import nnUNetDataLoader2D
from nnunetv2.training.dataloading.data_loader_3d import nnUNetDataLoader3D_MRCT, nnUNetDataLoader3D_MRCT_mask

from nnunetv2.utilities.helpers import empty_cache, dummy_context
from nnunetv2.training.nnUNetTrainer.variants.network_architecture.nnUNetTrainerMRCT import nnUNetTrainerMRCT_track
from batchgenerators.transforms.abstract_transforms import AbstractTransform, Compose
from batchgenerators.transforms.utility_transforms import RemoveLabelTransform, RenameTransform, NumpyToTensor
from nnunetv2.training.data_augmentation.custom_transforms.cascade_transforms import MoveSegAsOneHotToData, \
    ApplyRandomBinaryOperatorTransform, RemoveRandomConnectedComponentFromOneHotEncodingTransform

from torch import autocast
import numpy as np

from nnunetv2.training.dataloading.nnunet_dataset import nnUNetDatasetMask


class nnUNetTrainerMRCT_loss_masked(nnUNetTrainerMRCT_track):
    def __init__(
        self,
        plans: dict,
        configuration: str,
        fold: int,
        dataset_json: dict,
        unpack_dataset: bool = True,
        device: torch.device = torch.device("cuda")
    ):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.enable_deep_supervision = False
        self.num_iterations_per_epoch = 250
        self.num_epochs = 1000
        self.decoder_type = "standard" #["standard", "trilinear", "nearest"]  

    def _build_loss(self):
        # loss = myMSE()
        loss= myMaskedMSE()
        return loss
    
    def get_tr_and_val_datasets(self):
        # create dataset split
        tr_keys, val_keys = self.do_split()

        # load the datasets for training and validation. Note that we always draw random samples so we really don't
        # care about distributing training cases across GPUs.
        dataset_tr = nnUNetDatasetMask(self.preprocessed_dataset_folder, tr_keys,
                                   folder_with_segs_from_previous_stage=self.folder_with_segs_from_previous_stage,
                                   num_images_properties_loading_threshold=0)
        dataset_val = nnUNetDatasetMask(self.preprocessed_dataset_folder, val_keys,
                                    folder_with_segs_from_previous_stage=self.folder_with_segs_from_previous_stage,
                                    num_images_properties_loading_threshold=0)
        return dataset_tr, dataset_val
    
    def get_plain_dataloaders(self, initial_patch_size: Tuple[int, ...], dim: int):
        dataset_tr, dataset_val = self.get_tr_and_val_datasets()

        initial_patch_size=self.configuration_manager.patch_size
        dim=dim

        if dim == 2:
            assert(0) # todo
        else:
            dl_tr = nnUNetDataLoader3D_MRCT_mask(dataset_tr, self.batch_size,
                                       initial_patch_size,
                                       self.configuration_manager.patch_size,
                                       self.label_manager,
                                       oversample_foreground_percent=self.oversample_foreground_percent,
                                       sampling_probabilities=None, pad_sides=None)
            dl_val = nnUNetDataLoader3D_MRCT_mask(dataset_val, self.batch_size,
                                        self.configuration_manager.patch_size,
                                        self.configuration_manager.patch_size,
                                        self.label_manager,
                                        oversample_foreground_percent=self.oversample_foreground_percent,
                                        sampling_probabilities=None, pad_sides=None)
        return dl_tr, dl_val

    def train_step(self, batch: dict) -> dict:
        data = batch['data']
        target = batch['target']
        mask = batch['mask']

        data = data.to(self.device, non_blocking=True)
        mask = mask.to(self.device, non_blocking=True)
        if isinstance(target, list):
            target = [i.to(self.device, non_blocking=True) for i in target]
        else:
            target = target.to(self.device, non_blocking=True)

        self.optimizer.zero_grad(set_to_none=True)
        # Autocast can be annoying
        # If the device_type is 'cpu' then it's slow as heck and needs to be disabled.
        # If the device_type is 'mps' then it will complain that mps is not implemented, even if enabled=False is set. Whyyyyyyy. (this is why we don't make use of enabled=False)
        # So autocast will only be active if we have a cuda device.
        with autocast(self.device.type, enabled=True) if self.device.type == 'cuda' else dummy_context():
            output = self.network(data)
            # print(self.network)
            # assert(0)

            # del data
            # l = self.loss(output, target)
            l = self.loss(output, target, mask=mask)

        if self.grad_scaler is not None:
            self.grad_scaler.scale(l).backward()
            self.grad_scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            l.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.optimizer.step()
        return {'loss': l.detach().cpu().numpy()}
    

    def validation_step(self, batch: dict) -> dict:
        data = batch['data']
        target = batch['target']
        mask = batch['mask']

        data = data.to(self.device, non_blocking=True)
        mask = mask.to(self.device, non_blocking=True)
        if isinstance(target, list):
            target = [i.to(self.device, non_blocking=True) for i in target]
        else:
            target = target.to(self.device, non_blocking=True)

        with autocast(self.device.type, enabled=True) if self.device.type == 'cuda' else dummy_context():
            output = self.network(data)
            # torch.save(data, "data")
            # torch.save(output, "output")
            # torch.save(target, "target")

            del data
            l = self.loss(output, target, mask=mask)

        return {'loss': l.detach().cpu().numpy(), 'tp_hard': 0, 'fp_hard': 0, 'fn_hard': 0}

    
    @staticmethod
    def get_validation_transforms(
            deep_supervision_scales: Union[List, Tuple, None],
            is_cascaded: bool = False,
            foreground_labels: Union[Tuple[int, ...], List[int]] = None,
            regions: List[Union[List[int], Tuple[int, ...], int]] = None,
            ignore_label: int = None,
    ) -> AbstractTransform:
        val_transforms = []
        val_transforms.append(RemoveLabelTransform(-1, 0))

        val_transforms.append(RenameTransform('seg', 'target', True))

        val_transforms.append(NumpyToTensor(['data', 'target', 'mask'], 'float'))
        val_transforms = Compose(val_transforms)
        return val_transforms
    
    @staticmethod
    def get_training_transforms(patch_size: Union[np.ndarray, Tuple[int]],
                                rotation_for_DA: dict,
                                deep_supervision_scales: Union[List, Tuple, None],
                                mirror_axes: Tuple[int, ...],
                                do_dummy_2d_data_aug: bool,
                                order_resampling_data: int = 1,
                                order_resampling_seg: int = 0,
                                border_val_seg: int = -1,
                                use_mask_for_norm: List[bool] = None,
                                is_cascaded: bool = False,
                                foreground_labels: Union[Tuple[int, ...], List[int]] = None,
                                regions: List[Union[List[int], Tuple[int, ...], int]] = None,
                                ignore_label: int = None) -> AbstractTransform:
        return nnUNetTrainerMRCT_loss_masked.get_validation_transforms(deep_supervision_scales, is_cascaded, foreground_labels,
                                              regions, ignore_label)



