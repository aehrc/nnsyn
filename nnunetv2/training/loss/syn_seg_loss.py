import torch
from torch import nn
import numpy as np

from nnunetv2.training.loss.dice import SoftDiceLoss, MemoryEfficientSoftDiceLoss
from nnunetv2.training.loss.compound_losses import DC_and_CE_loss, DC_and_BCE_loss
from nnunetv2.utilities.helpers import softmax_helper_dim1
from batchgenerators.utilities.file_and_folder_operations import load_json,join
from nnunetv2.utilities.label_handling.label_handling import determine_num_input_channels
from nnunetv2.training.nnUNetTrainer.nnUNetTSTrainer import nnUNetTSTrainer
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager, ConfigurationManager
from acvl_utils.cropping_and_padding.padding import pad_nd_image
from nnunetv2.training.loss.mse import myMSE, myMaskedMSE





class SynSegLoss(nn.Module):
    def __init__(self, region: str, image_loss_weight: float = 0.5):
        """
        Initializes the SynSegLoss module.
        region can be "AB", "HN", or "TH" to specify the anatomical region.
        """
        super(SynSegLoss, self).__init__()
        self.region = region

        # load the trained segmentor model
        self.seg_model, self.seg_model_info = self._load_trained_segmentor(region)
        self.seg_model.eval()
        for param in self.seg_model.parameters(): 
            param.requires_grad = False
        self.seg_model.to(device='cuda', dtype=torch.float16)

        # specify the segmentation loss
        self.seg_loss = DC_and_BCE_loss({}, {'batch_dice': True,
                                   'smooth': 1e-5, 'do_bg': False, 'ddp': False}, weight_ce=1, weight_dice=1,
                                  use_ignore_label=False, dice_class=MemoryEfficientSoftDiceLoss)
        # self.seg_loss = SoftDiceLoss(apply_nonlin=softmax_helper_dim1, batch_dice=True, do_bg=False, smooth=0, ddp=False)

        # specify the image similarity loss
        self.image_loss = myMaskedMSE()
        self.image_loss_weight = image_loss_weight  # You can adjust this weight as needed

        # logging current losses 
        self.cur_seg_loss = 0.0
        self.cur_img_loss = 0.0
    
    def _load_trained_segmentor(self, region: str):
        segmentor_training_output_dir = {
            "AB": "/datasets/work/hb-synthrad2023/work/synthrad2025/bw_workplace/data/nnunet_struct/results/Dataset800_SEGMENTATION_synthrad2025_task1_CT_AB_aligned_to_Dataset261/nnUNetTrainer__nnUNetResEncUNetLPlans_Dataset261__3d_fullres",
            "HN": "todo",
            "TH": "todo"
        }
        # model_training_output_dir = '/datasets/work/hb-synthrad2023/work/synthrad2025/bw_workplace/ref/evaluation/.totalsegmentator/nnunet/results/Dataset297_TotalSegmentator_total_3mm_1559subj/nnUNetTrainer_4000epochs_NoMirroring__nnUNetPlans__3d_fullres'
        model_training_output_dir = segmentor_training_output_dir[region]
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
        # trainer_class = recursive_find_python_class(join(nnunetv2.__path__[0], "training", "nnUNetTrainer"),
        #                                             "nnUNetTSTrainer", 'nnunetv2.training.nnUNetTSTrainer.nnUNetTSTrainer')
        trainer_class = nnUNetTSTrainer

        network = trainer_class.build_network_architecture(
            configuration_manager.network_arch_class_name,
            configuration_manager.network_arch_init_kwargs,
            configuration_manager.network_arch_init_kwargs_req_import,
            num_input_channels,
            plans_manager.get_label_manager(dataset_json).num_segmentation_heads,
            enable_deep_supervision=False
        )
        network.load_state_dict(checkpoint['network_weights'])

        network_info = {
            "num_classes": plans_manager.get_label_manager(dataset_json).num_segmentation_heads, 
            "patch_size": configuration_manager.patch_size,
        }

        return network, network_info


    def _get_gt_segmentation(self, mask: torch.Tensor) -> torch.Tensor:
        gt_segmentation = mask.clone()
        gt_segmentation[gt_segmentation < 0] = 0  # Ensure no negative values in the mask
        # Convert to one-hot encoding
        gt_segmentation_onehot = torch.zeros((gt_segmentation.shape[0], self.seg_model_info["num_classes"], gt_segmentation.shape[2], gt_segmentation.shape[3], gt_segmentation.shape[4]), device=gt_segmentation.device)
        gt_segmentation_onehot.scatter_(1, gt_segmentation.long(), 1)
        return gt_segmentation_onehot
    
    def _get_real_mask(self, mask: torch.Tensor) -> torch.Tensor:
        """
        Returns the real mask for the segmentation task.
        """
        # Here we assume that the mask is already in the correct format
        # If you need to process it further, you can add that logic here
        mask[mask!= 0] = 1  # Convert all non-zero values to 1
        return mask
    
    # def select_valid_labels(self, output: torch.Tensor) -> torch.Tensor:
    #     return output[:, self.seg_valid_labels, :, :, :]

    def forward(self, output: torch.Tensor, target: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        # compute the segmentation loss

        gt_segmentation = self._get_gt_segmentation(mask)
        pred = self.seg_model(output)
        # print('gt_segmentation shape:', gt_segmentation.shape, gt_segmentation.max(), gt_segmentation.min())
        # print('pred shape:', pred.shape, pred.max(), pred.min())
        seg_loss = self.seg_loss(pred, gt_segmentation)
        # compute the image similarity loss
        real_mask = self._get_real_mask(mask)
        img_loss = self.image_loss(output, target, mask=real_mask)

        # log the current losses
        self.cur_seg_loss = seg_loss.detach().cpu().numpy()
        self.cur_img_loss = img_loss.detach().cpu().numpy()

        return seg_loss * (1 - self.image_loss_weight) + img_loss * self.image_loss_weight

if __name__ == "__main__":
    # Example usage
    region = "AB"  # or "HN", "TH"
    syn_seg_loss = SynSegLoss(region)
    
    # Dummy data for testing
    output = torch.randn(2, 1, 40, 192, 192)  # Example output from a model
    target = torch.randn(2, 1, 40, 192, 192)  # Example ground truth mask
    mask = torch.randint(0, 62, (2, 1, 40, 192, 192))  # Example mask
    output = output.to(device='cuda', dtype=torch.float16)  # Move to GPU if available
    target = target.to(device='cuda', dtype=torch.float16)  # Move to GPU if available
    mask = mask.to(device='cuda', dtype=torch.float16)  # Move to GPU if available

    loss_value = syn_seg_loss(output, target, mask)
    print(f"Loss value: {loss_value.item()}")