import torch 
from nnunetv2.utilities.get_network_from_plans import get_network_from_plans 
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager 
# from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer 
from nnunetv2.paths import nnUNet_results 
from batchgenerators.utilities.file_and_folder_operations import load_json 
from nnunetv2.paths import nnUNet_preprocessed, nnUNet_results 
from batchgenerators.utilities.file_and_folder_operations import join
from nnunetv2.training.nnUNetTrainer.variants.network_architecture.nnUNetTrainerMRCT_zz_loss_masked_perception import nnUNetTrainerMRCT_loss_masked_perception_masked


# Replace with actual paths and configuration names as per your setup
nnUNet_preprocessed = '/datasets/work/hb-synthrad2023/work/synthrad2025/bw_workplace/data/nnunet_struct/preprocessed'
plans_path = join(nnUNet_preprocessed, 'Dataset540_synthrad2025_task2_CBCT_AB_pre_v2r_stitched_masked_both/nnUNetResEncUNetLPlans.json')
dataset_json_path = join(nnUNet_preprocessed, 'Dataset540_synthrad2025_task2_CBCT_AB_pre_v2r_stitched_masked_both/dataset.json')
configuration = '3d_fullres'  # Example configuration, replace with your actual configuration
fold = 0  # Example fold

# Load plans and dataset_json 
plans = load_json(plans_path) 
dataset_json = load_json(dataset_json_path)

# Initialize the PlansManager and ConfigurationManager 
plans_manager = PlansManager(plans) 
configuration_manager = plans_manager.get_configuration(configuration)

# Ensure nnUNet_preprocessed and nnUNet_results are not None
if nnUNet_preprocessed is None or nnUNet_results is None:
    raise ValueError("nnUNet_preprocessed and nnUNet_results must be set to valid paths.")

# Initialize the trainer
trainer = nnUNetTrainerMRCT_loss_masked_perception_masked(plans, configuration, fold, dataset_json)
# Initialize the network
trainer.initialize()

# Function to count parameters
def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params

# Get the number of parameters
total_params, trainable_params = count_parameters(trainer.network)
print(f'Total parameters in nnUNet model: {total_params}')
print(f'Trainable parameters in nnUNet model: {trainable_params}')