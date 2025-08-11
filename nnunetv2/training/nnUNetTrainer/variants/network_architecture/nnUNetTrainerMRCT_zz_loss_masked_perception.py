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
from nnunetv2.training.nnUNetTrainer.variants.network_architecture.nnUNetTrainerMRCT_zz_loss_masked import nnUNetTrainerMRCT_loss_masked
from nnunetv2.training.loss.syn_perception_loss import SynPerceptionLoss, SynPerceptionLoss_L2


class nnUNetTrainerMRCT_loss_masked_perception(nnUNetTrainerMRCT_loss_masked):
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
        region = self._get_region_name()
        loss= SynPerceptionLoss(region=region, image_loss_weight=0.5)
        return loss

class nnUNetTrainerMRCT_loss_masked_perception_L2(nnUNetTrainerMRCT_loss_masked):
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
        region = self._get_region_name()
        loss= SynPerceptionLoss_L2(region=region, image_loss_weight=0.5)
        return loss



