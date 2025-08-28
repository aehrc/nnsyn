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
from nnunetv2.utilities.get_network_from_plans import get_network_from_plans
from torch import autocast, nn
from nnunetv2.training.nnUNetTrainer.variants.network_architecture.nnUNetTrainerMRCT import nnUNetTrainerMRCT


from torch import autocast

import os
import aim
import json
import numpy as np
from nnunetv2.analysis.revert_normalisation import get_ct_normalisation_values, revert_normalisation
from nnunetv2.analysis.result_analysis_brain import FinalValidationResults, ValidationResults




class nnUNetTrainerMRCT_track_brain(nnUNetTrainerMRCT):
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

    # def _get_region_name(self) -> str:
    #     """
    #     Returns the region name for the loss function.
    #     """
    #     if '_AB_' in self.plans_manager.dataset_name:
    #         return 'AB'
    #     elif '_TH_' in self.plans_manager.dataset_name:
    #         return 'TH'
    #     elif '_HN_' in self.plans_manager.dataset_name:
    #         return 'HN'
    #     else:
    #         raise ValueError("Unknown region in dataset name: {}".format(self.plans_manager.dataset_name))

    # def _get_task_name(self) -> str:
    #     """
    #     Returns the task name for the loss function.
    #     """
    #     if '_task1_' in self.plans_manager.dataset_name:
    #         return '1'
    #     elif '_task2_' in self.plans_manager.dataset_name:
    #         return '2'
    #     else:
    #         raise ValueError("Unknown task in dataset name: {}".format(self.plans_manager.dataset_name))
    
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
        # dataset_id = int(self.plans_manager.dataset_name.split('_')[0].replace('Dataset', ''))
        # if self.aim_run['task'] == '1':
        #     ct_plan_path = f"{nnUNet_preprocessed}/{self.plans_manager.dataset_name.replace('_MR_', '_CT_').replace(str(dataset_id), str(dataset_id+1))}/{self.plans_manager.plans_name}.json"
        # elif self.aim_run['task'] == '2':
        #     ct_plan_path = f"{nnUNet_preprocessed}/{self.plans_manager.dataset_name.replace('_CBCT_', '_CT_').replace(str(dataset_id), str(dataset_id+1))}/{self.plans_manager.plans_name}.json"
        # else:
        #     raise ValueError(f"Unknown task {self.aim_run['task']} in dataset name {self.plans_manager.dataset_name}. Cannot determine CT plan path.")
        # ct_mean, ct_std = get_ct_normalisation_values(ct_plan_path)

        pred_path = join(self.output_folder, 'validation')
        # pred_path_revert_norm = pred_path + "_revert_norm"
        # mask_path = join(nnUNet_preprocessed, self.plans_manager.dataset_name, 'masks_real')
        mask_path = mask_path.replace('masks_real', 'masks') if not os.path.isdir(mask_path) else mask_path
        print('=====> mask_path:', mask_path)
        # revert_normalisation(pred_path, ct_mean, ct_std, save_path=pred_path_revert_norm, mask_path=mask_path, mask_outside_value=-1000)

        # compute metrics
        gt_path = join(nnUNet_preprocessed, self.plans_manager.dataset_name, 'gt_target')
        src_path = join(nnUNet_raw, self.plans_manager.dataset_name, 'imagesTr')
        # gt_segmentation_path = join(nnUNet_preprocessed, self.plans_manager.dataset_name, 'gt_target_segmentation_ts')
        # gt_segmentation_path = gt_segmentation_path if os.path.isdir(gt_segmentation_path) else None
        
        ts = FinalValidationResults(pred_path, gt_path, mask_path, src_path, gt_segmentation_path=None)
        dict_metric = ts.process_patients_mp()
        self.aim_run['final_mae_mean'] = np.round(dict_metric['mae']['mean'], decimals=4)
        self.aim_run['final_psnr_mean'] = np.round(dict_metric['psnr']['mean'], decimals=4)
        self.aim_run['final_ms_ssim_mean'] = np.round(dict_metric['ms_ssim']['mean'], decimals=4)

        # print to console
        self.print_to_log_file(f'Final MAE: {self.aim_run["final_mae_mean"]}')
        self.print_to_log_file(f'Final PSNR: {self.aim_run["final_psnr_mean"]}')
        self.print_to_log_file(f'Final MS-SSIM: {self.aim_run["final_ms_ssim_mean"]}')
        # if gt_segmentation_path:
        #     self.aim_run['final_DICE_mean'] = np.round(dict_metric['DICE']['mean'], decimals=4)
        #     self.aim_run['final_HD95_mean'] = np.round(dict_metric['HD95']['mean'], decimals=4)
        #     self.print_to_log_file(f'Final DICE: {self.aim_run["final_DICE_mean"]}')
        #     self.print_to_log_file(f'Final HD95: {self.aim_run["final_HD95_mean"]}')

        # save dict_metric results to pred_path as summary.json
        summary_path = join(pred_path, 'summary.json')
        with open(summary_path, 'w') as f:
            json.dump(dict_metric, f, indent=4)
                
        self.print_to_log_file('Final validation completed. Results saved to Aim. Exiting the process.')
        sys.exit(0)  # Exit the process after training ends


        
    
    def perform_actual_validation_epoch(self, save_probabilities: bool = False):
        super().perform_actual_validation()
        self.set_deep_supervision_enabled(False)
        self.network.train()
        # revert normalisation
        # dataset_id = int(self.plans_manager.dataset_name.split('_')[0].replace('Dataset', ''))
        # if self.aim_run['task'] == '1':
        #     ct_plan_path = f"{nnUNet_preprocessed}/{self.plans_manager.dataset_name.replace('_MR_', '_CT_').replace(str(dataset_id), str(dataset_id+1))}/{self.plans_manager.plans_name}.json"
        # elif self.aim_run['task'] == '2':
        #     ct_plan_path = f"{nnUNet_preprocessed}/{self.plans_manager.dataset_name.replace('_CBCT_', '_CT_').replace(str(dataset_id), str(dataset_id+1))}/{self.plans_manager.plans_name}.json"
        # else:
        #     raise ValueError(f"Unknown task {self.aim_run['task']} in dataset name {self.plans_manager.dataset_name}. Cannot determine CT plan path.")
        
        # ct_mean, ct_std = get_ct_normalisation_values(ct_plan_path)

        pred_path = join(self.output_folder, 'validation')
        # pred_path_revert_norm = pred_path + "_revert_norm"
        # use masked image after revert normalisation
        mask_path = join(nnUNet_preprocessed, self.plans_manager.dataset_name, 'masks_real')
        mask_path = mask_path.replace('masks_real', 'masks') if not os.path.isdir(mask_path) else mask_path
        print('=====> mask_path:', mask_path)
        # revert_normalisation(pred_path, ct_mean, ct_std, save_path=pred_path_revert_norm, mask_path=mask_path, mask_outside_value=-1000)

        # compute metrics
        gt_path = join(nnUNet_preprocessed, self.plans_manager.dataset_name, 'gt_target')
        # gt_segmentation_path = join(nnUNet_preprocessed, self.plans_manager.dataset_name, 'gt_target_segmentation_ts')
        gt_segmentation_path =  None
        # gt_segmentation_path = join(nnUNet_preprocessed, self.plans_manager.dataset_name, 'gt_target_segmentation') if os.path.isdir(join(nnUNet_preprocessed, self.plans_manager.dataset_name, 'gt_segmentation')) else None
        src_path = join(nnUNet_raw, self.plans_manager.dataset_name, 'imagesTr')
        # print("gt_segmentation_path: ", gt_segmentation_path)
        ts = ValidationResults(pred_path, gt_path, mask_path, src_path, gt_segmentation_path=gt_segmentation_path)
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
        # if gt_segmentation_path:
        #     self.aim_run.track(np.round(dict_metric['DICE']['mean'], decimals=4), \
        #                        name="DICE_mean", context={"type": 'metrics'}, step=self.current_epoch)
        #     self.aim_run.track(np.round(dict_metric['HD95']['mean'], decimals=4), \
        #                        name="HD95_mean", context={"type": 'metrics'}, step=self.current_epoch)
        #     self.print_to_log_file(f'DICE: {np.round(dict_metric["DICE"]["mean"], decimals=4)}')
        #     self.print_to_log_file(f'HD95: {np.round(dict_metric["HD95"]["mean"], decimals=4)}')

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



        


