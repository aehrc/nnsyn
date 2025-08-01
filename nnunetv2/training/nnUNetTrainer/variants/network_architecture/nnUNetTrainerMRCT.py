import sys
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
import torch
from typing import Union, Tuple, List

from batchgenerators.transforms.abstract_transforms import AbstractTransform

import numpy as np
from nnunetv2.training.loss.mse import myMSE

from nnunetv2.training.dataloading.data_loader_2d import nnUNetDataLoader2D
from nnunetv2.training.dataloading.data_loader_3d import nnUNetDataLoader3D_MRCT

from torch import distributed as dist
from nnunetv2.utilities.collate_outputs import collate_outputs

from time import time, sleep
from batchgenerators.utilities.file_and_folder_operations import join, load_json, isfile, save_json, maybe_mkdir_p
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
import multiprocessing
from nnunetv2.inference.sliding_window_prediction import compute_gaussian
from nnunetv2.training.dataloading.nnunet_dataset import nnUNetDataset
from nnunetv2.utilities.file_path_utilities import check_workers_alive_and_busy
from nnunetv2.utilities.helpers import empty_cache, dummy_context
from nnunetv2.configuration import ANISO_THRESHOLD, default_num_processes
from nnunetv2.utilities.label_handling.label_handling import convert_labelmap_to_one_hot, determine_num_input_channels
from nnunetv2.evaluation.evaluate_predictions import compute_metrics_on_folder
from nnunetv2.paths import nnUNet_preprocessed, nnUNet_results, nnUNet_raw
from nnunetv2.inference.export_prediction import export_prediction_from_logits, resample_and_save
import warnings


from torch import autocast


class nnUNetTrainerMRCT(nnUNetTrainer):
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
        loss = myMSE()
        return loss

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
        return nnUNetTrainer.get_validation_transforms(deep_supervision_scales, is_cascaded, foreground_labels,
                                                       regions, ignore_label)

    def configure_rotation_dummyDA_mirroring_and_inital_patch_size(self):
        # we need to disable mirroring here so that no mirroring will be applied in inferene!
        rotation_for_DA, do_dummy_2d_data_aug, initial_patch_size, mirror_axes = \
            super().configure_rotation_dummyDA_mirroring_and_inital_patch_size()
        mirror_axes = None
        self.inference_allowed_mirroring_axes = None
        return rotation_for_DA, do_dummy_2d_data_aug, initial_patch_size, mirror_axes

    def train_step(self, batch: dict) -> dict:
        data = batch['data']
        target = batch['target']

        data = data.to(self.device, non_blocking=True)
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
            l = self.loss(output, target)

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
    
    def get_plain_dataloaders(self, initial_patch_size: Tuple[int, ...], dim: int):
        dataset_tr, dataset_val = self.get_tr_and_val_datasets()

        initial_patch_size=self.configuration_manager.patch_size
        dim=dim

        if dim == 2:
            assert(0) # todo
            dl_tr = nnUNetDataLoader2D(dataset_tr, self.batch_size,
                                       initial_patch_size,
                                       self.configuration_manager.patch_size,
                                       self.label_manager,
                                       oversample_foreground_percent=self.oversample_foreground_percent,
                                       sampling_probabilities=None, pad_sides=None)
            dl_val = nnUNetDataLoader2D(dataset_val, self.batch_size,
                                        self.configuration_manager.patch_size,
                                        self.configuration_manager.patch_size,
                                        self.label_manager,
                                        oversample_foreground_percent=self.oversample_foreground_percent,
                                        sampling_probabilities=None, pad_sides=None)
        else:
            dl_tr = nnUNetDataLoader3D_MRCT(dataset_tr, self.batch_size,
                                       initial_patch_size,
                                       self.configuration_manager.patch_size,
                                       self.label_manager,
                                       oversample_foreground_percent=self.oversample_foreground_percent,
                                       sampling_probabilities=None, pad_sides=None)
            dl_val = nnUNetDataLoader3D_MRCT(dataset_val, self.batch_size,
                                        self.configuration_manager.patch_size,
                                        self.configuration_manager.patch_size,
                                        self.label_manager,
                                        oversample_foreground_percent=self.oversample_foreground_percent,
                                        sampling_probabilities=None, pad_sides=None)
        return dl_tr, dl_val

    def validation_step(self, batch: dict) -> dict:
        data = batch['data']
        target = batch['target']

        data = data.to(self.device, non_blocking=True)
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
            mse_loss = myMSE()
            l = mse_loss(output, target)

        return {'loss': l.detach().cpu().numpy(), 'tp_hard': 0, 'fp_hard': 0, 'fn_hard': 0}


    def on_validation_epoch_end(self, val_outputs: List[dict]):
        outputs_collated = collate_outputs(val_outputs)
        
        loss_here = np.mean(outputs_collated['loss'])

        # self.logger.log('mean_fg_dice', mean_fg_dice, self.current_epoch)
        # self.logger.log('dice_per_class_or_region', global_dc_per_class, self.current_epoch)
        self.logger.log('val_losses', loss_here, self.current_epoch)

    def on_epoch_end(self):
        # Log the end time of the epoch
        self.logger.log('epoch_end_timestamps', time(), self.current_epoch)

        # Logging train and validation loss
        self.print_to_log_file('train_loss', np.round(self.logger.my_fantastic_logging['train_losses'][-1], decimals=4))
        self.print_to_log_file('val_loss', np.round(self.logger.my_fantastic_logging['val_losses'][-1], decimals=4))
        
        # Log the duration of the epoch
        epoch_duration = self.logger.my_fantastic_logging['epoch_end_timestamps'][-1] - self.logger.my_fantastic_logging['epoch_start_timestamps'][-1]
        self.print_to_log_file(f"Epoch time: {np.round(epoch_duration, decimals=2)} s")

        # Checkpoint handling for best and periodic saves
        current_epoch = self.current_epoch
        if (current_epoch + 1) % self.save_every == 0 and current_epoch != (self.num_epochs - 1):
            self.save_checkpoint(join(self.output_folder, 'checkpoint_latest.pth'))

        best_metric = 'val_losses'  # Example metric, adjust based on actual usage
        if self._best_ema is None or self.logger.my_fantastic_logging[best_metric][-1] < self._best_ema:
            self._best_ema = self.logger.my_fantastic_logging[best_metric][-1]
            self.print_to_log_file(f"Yayy! New best EMA MSE: {np.round(self._best_ema, decimals=4)}")
            self.save_checkpoint(join(self.output_folder, 'checkpoint_best.pth'))

        if self.local_rank == 0:
            self.logger.plot_progress_png(self.output_folder)

        # Increment the epoch counter
        self.current_epoch += 1

import os
import aim
import json
import numpy as np
from nnunetv2.analysis.revert_normalisation import get_ct_normalisation_values, revert_normalisation
from nnunetv2.analysis.result_analysis import FinalValidationResults, ValidationResults




class nnUNetTrainerMRCT_track(nnUNetTrainerMRCT):
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
        self._init_aim()

    def _init_aim(self):
        # init aim
        repo_path = join(os.environ['nnUNet_results'], 'runs_aim')
        experiment = '//'.join(self.output_folder_base.split(os.sep)[-2:])
        # load previous run hash if exists
        if os.path.exists(join(self.output_folder, 'debug_custom.json')):
            with open(join(self.output_folder, 'debug_custom.json'), 'r') as f:
                self.debug_custom = json.load(f)
            # initialise aim system
            try:
                run_hash = self.debug_custom['run_hash']
                self.aim_run = aim.Run(repo=repo_path,
                                    experiment=experiment, run_hash=run_hash)
                print(f'AIM. Old run_hash: {run_hash}')

            except:
                print('AIM. Old run_hash cannot be found. Close the old process and try again')
            try:
                os.system(f'aim runs --repo {repo_path} close {run_hash} --yes')
                self.aim_run = aim.Run(repo=repo_path,
                                    experiment=experiment, run_hash=run_hash)  
                print('AIM. Old run_hash was found and restarted.')                 
            except:
                self.aim_run = aim.Run(repo=repo_path,
                                        experiment=experiment)
                print('AIM. Old run_hash cannot be found. Start a new one. ')
        # else start a new run
        else:
            self.aim_run = aim.Run(repo=repo_path,
                                        experiment=experiment)
            run_hash = self.aim_run.hash
            self.debug_custom = dict()
            self.debug_custom['run_hash'] = run_hash
            with open(join(self.output_folder, 'debug_custom.json'), 'wt') as file:
                json.dump(self.debug_custom, file)
            print('AIM. Old run_hash cannot be found. Start a new one. ')

        # log some initial information
        self.aim_run['fold'] = self.fold
        self.aim_run['dataset_name'] = self.plans_manager.dataset_name
        self.aim_run['dataset_id'] = self.plans_manager.dataset_name.split('_')[0].replace('Dataset', '')
        self.aim_run['configuration'] = self.configuration_name
        self.aim_run['plan'] = self.plans_manager.plans_name
        self.aim_run['trainer'] = self.__class__.__name__
        self.aim_run['num_epochs'] = self.num_epochs
        self.aim_run['current_epoch'] = self.current_epoch
        self.aim_run['job_id'] = os.environ['SLURM_JOB_ID'] if 'SLURM_JOB_ID' in os.environ else 'local_run'
        # assign region based on dataset name
        if '_AB_' in self.plans_manager.dataset_name:
            self.aim_run['region'] = 'AB'
        elif '_TH_' in self.plans_manager.dataset_name:
            self.aim_run['region'] = 'TH'
        elif '_HN_' in self.plans_manager.dataset_name:
            self.aim_run['region'] = 'HN' 
        else:
            self.aim_run['region'] = 'Unknown'

        # assign task based on dataset name
        if '_task1_' in self.plans_manager.dataset_name:
            self.aim_run['task'] = '1'
        elif '_task2_' in self.plans_manager.dataset_name:
            self.aim_run['task'] = '2'
        else:
            self.aim_run['task'] = 'Unknown'

    def on_epoch_end(self):
        super().on_epoch_end()
        self.aim_run.track(np.round(self.logger.my_fantastic_logging['train_losses'][-1], decimals=4), \
                           name="train_loss", context={"type": 'loss'}, step=self.current_epoch + 1)

        self.aim_run.track(np.round(self.logger.my_fantastic_logging['val_losses'][-1], decimals=4), \
                           name="val_loss", context={"type": 'loss'}, step=self.current_epoch + 1)
        
        epoch_duration = self.logger.my_fantastic_logging['epoch_end_timestamps'][-1] - self.logger.my_fantastic_logging['epoch_start_timestamps'][-1]
        self.aim_run.track(np.round(epoch_duration, decimals=2), name='epoch_time', step=self.current_epoch + 1)
        self.aim_run['current_epoch'] = self.current_epoch + 1
        
        # perform validation every 100 epochs
        # print(self.current_epoch, self.num_epochs, actual_validation_every, self.current_epoch % actual_validation_every == 0)
        actual_validation_every = 100
        if (self.current_epoch + 1) % actual_validation_every == 0 or self.current_epoch in [10, 20, 50]:
            # perform actual validation
            cur_time = time()
            self.print_to_log_file(f'=========> Validation start', also_print_to_console=True)
            self.perform_actual_validation_epoch()
            self.print_to_log_file(f'<========= validation used time: {time() - cur_time}', also_print_to_console=True)


    def perform_actual_validation(self, save_probabilities: bool = False):
        """
        This method is called at the end of all training.
        It performs the actual validation, computes metrics, and logs them using Aim.
        """
        super().perform_actual_validation()
        self.set_deep_supervision_enabled(False)
        self.network.train()
        # revert normalisation
        dataset_id = int(self.plans_manager.dataset_name.split('_')[0].replace('Dataset', ''))
        ct_plan_path = f"{nnUNet_preprocessed}/{self.plans_manager.dataset_name.replace('_MR_', '_CT_').replace(str(dataset_id), str(dataset_id+1))}/{self.plans_manager.plans_name}.json"
        ct_mean, ct_std = get_ct_normalisation_values(ct_plan_path)

        pred_path = join(self.output_folder, 'validation')
        pred_path_revert_norm = pred_path + "_revert_norm"
        revert_normalisation(pred_path, ct_mean, ct_std, save_path=pred_path_revert_norm)

        # compute metrics
        gt_path = join(nnUNet_preprocessed, self.plans_manager.dataset_name, 'gt_target')
        mask_path = join(nnUNet_preprocessed, self.plans_manager.dataset_name, 'masks')
        src_path = join(nnUNet_raw, self.plans_manager.dataset_name, 'imagesTr')
        ts = FinalValidationResults(pred_path_revert_norm, gt_path, mask_path, src_path)
        dict_metric = ts.process_patients_mp()
        self.aim_run['final_mae_mean'] = np.round(dict_metric['mae']['mean'], decimals=4)
        self.aim_run['final_psnr_mean'] = np.round(dict_metric['psnr']['mean'], decimals=4)
        self.aim_run['final_ms_ssim_mean'] = np.round(dict_metric['ms_ssim']['mean'], decimals=4)

        # print to console
        self.print_to_log_file(f'Final MAE: {self.aim_run["final_mae_mean"]}')
        self.print_to_log_file(f'Final PSNR: {self.aim_run["final_psnr_mean"]}')
        self.print_to_log_file(f'Final MS-SSIM: {self.aim_run["final_ms_ssim_mean"]}')
                
        self.print_to_log_file('Final validation completed. Results saved to Aim. Exiting the process.')
        sys.exit(0)  # Exit the process after training ends


        
    
    def perform_actual_validation_epoch(self, save_probabilities: bool = False):
        super().perform_actual_validation()
        self.set_deep_supervision_enabled(False)
        self.network.train()
        # revert normalisation
        dataset_id = int(self.plans_manager.dataset_name.split('_')[0].replace('Dataset', ''))
        ct_plan_path = f"{nnUNet_preprocessed}/{self.plans_manager.dataset_name.replace('_MR_', '_CT_').replace(str(dataset_id), str(dataset_id+1))}/{self.plans_manager.plans_name}.json"
        ct_mean, ct_std = get_ct_normalisation_values(ct_plan_path)

        pred_path = join(self.output_folder, 'validation')
        pred_path_revert_norm = pred_path + "_revert_norm"
        revert_normalisation(pred_path, ct_mean, ct_std, save_path=pred_path_revert_norm)

        # compute metrics
        gt_path = join(nnUNet_preprocessed, self.plans_manager.dataset_name, 'gt_target')
        mask_path = join(nnUNet_preprocessed, self.plans_manager.dataset_name, 'masks')
        src_path = join(nnUNet_raw, self.plans_manager.dataset_name, 'imagesTr')
        ts = ValidationResults(pred_path_revert_norm, gt_path, mask_path, src_path)
        dict_metric = ts.process_patients_mp()
        ts.aim_log_one_patient(self.aim_run, self.current_epoch, max_images=4)
        # log metrics using aim
        self.aim_run.track(np.round(dict_metric['mae']['mean'], decimals=4), \
                           name="mae_mean", context={"type": 'metrics'}, step=self.current_epoch)
        self.aim_run.track(np.round(dict_metric['psnr']['mean'], decimals=4), \
                           name="psnr_mean", context={"type": 'metrics'}, step=self.current_epoch)
        self.aim_run.track(np.round(dict_metric['ms_ssim']['mean'], decimals=4), \
                           name="ms_ssim_mean", context={"type": 'metrics'}, step=self.current_epoch)
        
        # print to console
        self.print_to_log_file(f'MAE: {np.round(dict_metric["mae"]["mean"], decimals=4)}')
        self.print_to_log_file(f'PSNR: {np.round(dict_metric["psnr"]["mean"], decimals=4)}')
        self.print_to_log_file(f'MS-SSIM: {np.round(dict_metric["ms_ssim"]["mean"], decimals=4)}')

    def init_call_resume(self):
        import signal
        from subprocess import call

        def call_resume():
            print('Close aim tracking')
            self.aim_run.close()

            job_id = os.environ['SLURM_JOB_ID']
            cmd = f'scontrol requeue {job_id}'
            print(f'\nRequeing job {job_id}...')
            result = call(cmd, shell=True)
            if result == 0:
                print(f'Requeued job {job_id}.')
            else:
                print('Requeue failed...')

        def sig_handler(signum, frame):
            print(f'Caught signal: {signum} - requeuing the job')
            ### specify ending condition here: 
            if self.current_epoch < self.num_epochs:
                call_resume()
                print('Signal sigusr1. Requeue the job.')
            else:
                print('Reaching the No. epoch, terminates')

        def term_handler(signum, frame):
            print(f'Caught signal: {signum} - terminating the process')
            self.aim_run.close()
            if self.current_epoch == self.num_epochs:
                try:
                    os.system(f"scancel {os.environ['SLURM_JOB_ID']}") # fix job cancellation
                except:
                    print('Cannot cancel the job. Exit the process.')
            sys.exit(0)
            print('Signal sigterm. Close Aim. cancel the job and exit. ')
            # exit the process
        # start
        print('Setting signal to automatically requeue the job before timeout.')
        signal.signal(signal.SIGUSR1, sig_handler)
        signal.signal(signal.SIGTERM, term_handler)
        print('Start listening: waiting for signal')


class nnUNetTrainerMRCT_1500epochs(nnUNetTrainerMRCT_track):
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
        self.num_epochs = 1500



        


