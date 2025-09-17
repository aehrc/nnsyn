import torch
from typing import Union, Tuple, List
import numpy as np
from nnunetv2.training.loss.syn_perception_loss import SynPerceptionLoss, SynPerceptionLoss_L2
from nnunetv2.utilities.collate_outputs import collate_outputs
from nnunetv2.training.nnUNetTrainer.variants.nnsyn.nnUNetTrainer_nnsyn_loss_masked import nnUNetTrainer_nnsyn_loss_masked, nnUNetTrainer_nnsyn_loss_masked_track
from nnunetv2.training.nnUNetTrainer.variants.nnsyn.nnsyn_loss_map import MaskedAnatomicalPerceptionLoss


class nnUNetTrainer_nnsyn_loss_map(nnUNetTrainer_nnsyn_loss_masked):
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
        self.perception_masked = True
        # track losses 
        self.dataset_name_seg = self.plans_manager.plans['dataset_name_seg']
        self.logger.my_fantastic_logging['train_seg_loss'] = list()
        self.logger.my_fantastic_logging['train_img_loss'] = list()  
        self.logger.my_fantastic_logging['val_seg_loss'] = list()
        self.logger.my_fantastic_logging['val_img_loss'] = list()

    def _build_loss(self):
        # loss = myMSE()
        loss= MaskedAnatomicalPerceptionLoss(dataset_name_seg=self.dataset_name_seg, image_loss_weight=self.image_loss_weight, perception_masked=self.perception_masked)
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



