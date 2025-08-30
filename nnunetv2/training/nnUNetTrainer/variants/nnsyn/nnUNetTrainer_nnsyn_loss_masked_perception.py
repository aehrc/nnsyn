import torch
from typing import Union, Tuple, List
import numpy as np
from nnunetv2.training.loss.syn_perception_loss import SynPerceptionLoss, SynPerceptionLoss_L2
from nnunetv2.utilities.collate_outputs import collate_outputs
from nnunetv2.training.nnUNetTrainer.variants.nnsyn.nnUNetTrainer_nnsyn_loss_masked import nnUNetTrainer_nnsyn_loss_masked, nnUNetTrainer_nnsyn_loss_masked_track



class nnUNetTrainer_nnsyn_loss_masked_perception(nnUNetTrainer_nnsyn_loss_masked):
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
        self.image_loss_weight = 0.5  # default value, can be overridden in subclasses
        # track losses 
        self.logger.my_fantastic_logging['train_seg_loss'] = list()
        self.logger.my_fantastic_logging['train_img_loss'] = list()  
        self.logger.my_fantastic_logging['val_seg_loss'] = list()
        self.logger.my_fantastic_logging['val_img_loss'] = list()

    def _build_loss(self):
        # loss = myMSE()
        task = self._get_task_name()
        region = self._get_region_name()
        loss= SynPerceptionLoss(task=task, region=region, image_loss_weight=self.image_loss_weight)
        return loss
    
    # track losses
    def train_step(self, batch: dict) -> dict:
        outputs = super().train_step(batch)
        outputs['train_seg_loss'] = self.loss.cur_seg_loss
        outputs['train_img_loss'] = self.loss.cur_img_loss
        return outputs
    
    def validation_step(self, batch: dict) -> dict:
        outputs = super().validation_step(batch)
        outputs['val_seg_loss'] = self.loss.cur_seg_loss
        outputs['val_img_loss'] = self.loss.cur_img_loss
        return outputs

    def on_train_epoch_end(self, train_outputs: List[dict]):
        outputs = collate_outputs(train_outputs)
        self.logger.log('train_losses', np.mean(outputs['loss']), self.current_epoch)
        self.logger.log('train_seg_loss', np.mean(outputs['train_seg_loss']), self.current_epoch)
        self.logger.log('train_img_loss', np.mean(outputs['train_img_loss']), self.current_epoch)

    def on_validation_epoch_end(self, val_outputs: List[dict]):
        outputs = collate_outputs(val_outputs)
        self.logger.log('val_losses', np.mean(outputs['loss']), self.current_epoch)
        self.logger.log('val_seg_loss', np.mean(outputs['val_seg_loss']), self.current_epoch)
        self.logger.log('val_img_loss', np.mean(outputs['val_img_loss']), self.current_epoch)

    def on_epoch_end(self):
        super().on_epoch_end()
        self.aim_run.track(np.round(self.logger.my_fantastic_logging['train_seg_loss'][-1], decimals=4), \
                           name="train_seg_loss", context={"type": 'loss'}, step=self.current_epoch + 1)
        self.aim_run.track(np.round(self.logger.my_fantastic_logging['train_img_loss'][-1], decimals=4), \
                           name="train_img_loss", context={"type": 'loss'}, step=self.current_epoch + 1)
        self.aim_run.track(np.round(self.logger.my_fantastic_logging['val_seg_loss'][-1], decimals=4), \
                           name="val_seg_loss", context={"type": 'loss'}, step=self.current_epoch + 1)
        self.aim_run.track(np.round(self.logger.my_fantastic_logging['val_img_loss'][-1], decimals=4), \
                           name="val_img_loss", context={"type": 'loss'}, step=self.current_epoch + 1)

class nnUNetTrainer_nnsyn_loss_masked_perception_masked(nnUNetTrainer_nnsyn_loss_masked_perception):
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
        self.perception_masked = True
        self.image_loss_weight = 0.5  # default value, can be overridden in subclasses

    def _build_loss(self):
        # loss = myMSE()
        task = self._get_task_name()
        region = self._get_region_name()
        loss= SynPerceptionLoss(task=task, region=region, image_loss_weight=self.image_loss_weight, perception_masked=self.perception_masked)
        return loss
    
class nnUNetTrainer_nnsyn_loss_masked_perception_masked_track(nnUNetTrainer_nnsyn_loss_masked_perception_masked, nnUNetTrainer_nnsyn_loss_masked_track):
    pass


class nnUNetTrainer_nnsyn_loss_masked_perception_L2(nnUNetTrainer_nnsyn_loss_masked_perception):
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
        self.image_loss_weight = 0.5  # default value, can be overridden in subclasses

    def _build_loss(self):
        # loss = myMSE()
        task = self._get_task_name()
        region = self._get_region_name()
        loss= SynPerceptionLoss_L2(task=task, region=region, image_loss_weight=self.image_loss_weight)
        return loss

# class nnUNetTrainer_nnsyn_loss_masked_perception_L2_imglossweight0_7(nnUNetTrainer_nnsyn_loss_masked_perception_L2):
#     def __init__(
#         self,
#         plans: dict,
#         configuration: str,
#         fold: int,
#         dataset_json: dict,
#         unpack_dataset: bool = True,
#         device: torch.device = torch.device("cuda")
#     ):
#         super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
#         self.enable_deep_supervision = False
#         self.num_iterations_per_epoch = 250
#         self.num_epochs = 1000
#         self.decoder_type = "standard" #["standard", "trilinear", "nearest"]  

#         self.image_loss_weight = 0.7
        

# class nnUNetTrainerMRCT_loss_masked_perception_L2_SSIM(nnUNetTrainerMRCT_loss_masked_perception):
#     def __init__(
#         self,
#         plans: dict,
#         configuration: str,
#         fold: int,
#         dataset_json: dict,
#         unpack_dataset: bool = True,
#         device: torch.device = torch.device("cuda")
#     ):
#         super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
#         self.enable_deep_supervision = False
#         self.num_iterations_per_epoch = 250
#         self.num_epochs = 1000
#         self.decoder_type = "standard" #["standard", "trilinear", "nearest"]  

#     def _build_loss(self):
#         # loss = myMSE()
#         region = self._get_region_name()
#         loss= SynPerceptionLoss_L2_ssim(region=region, image_loss_weight=0.5)
#         return loss





