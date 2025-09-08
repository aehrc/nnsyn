import glob
import os
import torch
from torch import nn
import numpy as np
from typing import Union


from nnunetv2.training.loss.dice import SoftDiceLoss, MemoryEfficientSoftDiceLoss
from nnunetv2.training.loss.compound_losses import DC_and_CE_loss, DC_and_BCE_loss
from nnunetv2.utilities.helpers import softmax_helper_dim1
from batchgenerators.utilities.file_and_folder_operations import load_json,join
from nnunetv2.utilities.label_handling.label_handling import determine_num_input_channels
from nnunetv2.training.nnUNetTrainer.nnUNetTSTrainer import nnUNetTSTrainer
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager, ConfigurationManager
from acvl_utils.cropping_and_padding.padding import pad_nd_image
from nnunetv2.training.loss.mse import myMSE, myMaskedMSE
from nnunetv2.utilities.get_network_from_plans import get_network_from_plans_and_class
from nnunetv2.training.loss.unet import ResidualEncoderUNet
# from nnunetv2.training.loss.ssim_losses import SSIMLoss
from nnunetv2.utilities.dataset_name_id_conversion import maybe_convert_to_dataset_name



# https://github.com/richzhang/PerceptualSimilarity/blob/master/lpips/lpips.py
# https://github.com/Project-MONAI/GenerativeModels/blob/main/generative/losses/perceptual.py


class MaskedAnatomicalPerceptionLoss(nn.Module):
    def __init__(self, dataset_name_seg: int, image_loss_weight: float = 0.5, perception_masked=False):
        """
        Initializes the SynSegLoss module.
        region can be "AB", "HN", or "TH" to specify the anatomical region.
        """
        super(MaskedAnatomicalPerceptionLoss, self).__init__()

        # load the trained segmentor model
        self.seg_model, self.seg_model_info = self._load_trained_segmentor(dataset_name_seg)
        self.seg_model.eval()
        for param in self.seg_model.parameters(): 
            param.requires_grad = False
        self.seg_model.to(device='cuda', dtype=torch.float16)

        # specify the segmentation loss
        self.L1 = nn.L1Loss()
        self.perception_masked = perception_masked

        # specify the image similarity loss
        self.image_loss = myMaskedMSE()
        self.image_loss_weight = image_loss_weight  # You can adjust this weight as needed

        # logging current losses 
        self.cur_seg_loss = 0.0
        self.cur_img_loss = 0.0
    


    def _load_trained_segmentor(self, dataset_name_seg: Union[int, str]):
        # segmentor_training_output_dir = {
        #     "1":
        #         {"AB": "/datasets/work/hb-synthrad2023/work/synthrad2025/bw_workplace/data/nnunet_struct/results/Dataset800_SEGMENTATION_synthrad2025_task1_CT_AB_aligned_to_Dataset261/nnUNetTrainer__nnUNetResEncUNetLPlans_Dataset261__3d_fullres",
        #         "HN": "/datasets/work/hb-synthrad2023/work/synthrad2025/bw_workplace/data/nnunet_struct/results/Dataset801_SEGMENTATION_synthrad2025_task1_CT_HN_aligned_to_Dataset263/nnUNetTrainer__nnUNetResEncUNetLPlans_Dataset263__3d_fullres",
        #         "TH": "/datasets/work/hb-synthrad2023/work/synthrad2025/bw_workplace/data/nnunet_struct/results/Dataset802_SEGMENTATION_synthrad2025_task1_CT_TH_aligned_to_Dataset265/nnUNetTrainer__nnUNetResEncUNetLPlans_Dataset265__3d_fullres"},
        #     "2":
        #         {"AB": "/datasets/work/hb-synthrad2023/work/synthrad2025/bw_workplace/data/nnunet_struct/results/Dataset810_SEGMENTATION_synthrad2025_task2_CT_AB_aligned_to_Dataset541/nnUNetTrainer__nnUNetResEncUNetLPlans_Dataset541__3d_fullres",
        #         "HN": "/datasets/work/hb-synthrad2023/work/synthrad2025/bw_workplace/data/nnunet_struct/results/Dataset811_SEGMENTATION_synthrad2025_task2_CT_HN_aligned_to_Dataset543/nnUNetTrainer__nnUNetResEncUNetLPlans_Dataset543__3d_fullres",
        #         "TH": "/datasets/work/hb-synthrad2023/work/synthrad2025/bw_workplace/data/nnunet_struct/results/Dataset812_SEGMENTATION_synthrad2025_task2_CT_TH_aligned_to_Dataset545/nnUNetTrainer__nnUNetResEncUNetLPlans_Dataset545__3d_fullres"},
        # }
        # model_training_output_dir = '/datasets/work/hb-synthrad2023/work/synthrad2025/bw_workplace/ref/evaluation/.totalsegmentator/nnunet/results/Dataset297_TotalSegmentator_total_3mm_1559subj/nnUNetTrainer_4000epochs_NoMirroring__nnUNetPlans__3d_fullres'
        # model_training_output_dir = segmentor_training_output_dir[task][region]

        # /datasets/work/hb-synthrad2023/work/synthrad2025/bw_workplace/data/nnunet_struct/results/Dataset961_SEG_synthrad2025_task1_mri2ct_AB/nnUNetTrainer__nnUNetResEncUNetLPlans_Dataset960__3d_fullres
        # /datasets/work/hb-synthrad2023/work/synthrad2025/bw_workplace/data/nnunet_struct/results/Dataset960_synthrad2025_task1_mri2ct_AB/nnUNetTrainer_nnsyn_loss_masked_perception_masked_track__nnUNetResEncUNetLPlans__3d_fullres
        dataset_name_seg = maybe_convert_to_dataset_name(dataset_name_seg)
        segmentation_output_dir = glob.glob(os.path.join(os.environ['nnUNet_results'], dataset_name_seg, '*'))[0]
        assert len(glob.glob(os.path.join(os.environ['nnUNet_results'], dataset_name_seg, '*'))) == 1, f"Segmentation model output dir is more than one or empty: {segmentation_output_dir}"
        model_training_output_dir = segmentation_output_dir
        checkpoint_name = 'checkpoint_final.pth'
        if not os.path.exists(join(model_training_output_dir, f'fold_0', checkpoint_name)):
            checkpoint_name = 'checkpoint_best.pth'
            print('checkpoint_final.pth not found, using checkpoint_best.pth instead')
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

        network = get_network_from_plans_and_class(
            ResidualEncoderUNet,
            configuration_manager.network_arch_init_kwargs,
            configuration_manager.network_arch_init_kwargs_req_import,
            num_input_channels,
            plans_manager.get_label_manager(dataset_json).num_segmentation_heads,
        )
        network.load_state_dict(checkpoint['network_weights'])

        network_info = {
            "num_classes": plans_manager.get_label_manager(dataset_json).num_segmentation_heads, 
            "patch_size": configuration_manager.patch_size,
            "n_stages": configuration_manager.network_arch_init_kwargs['n_stages'],
        }

        return network, network_info

    def forward(self, output: torch.Tensor, target: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        if self.perception_masked:
            # apply the mask to the output and target tensors, and set the outside region to -1
            output = output * mask + (1 - mask) * -1
            target = target * mask + (1 - mask) * -1
        # compute the segmentation loss
        pred_outputs = self.seg_model(output)
        pred_gt_outputs = self.seg_model(target)

        perception_loss = 0
        layer_losses = []
        for i in range(self.seg_model_info['n_stages']):
            layer_loss = self.L1(self._normalize_tensor(pred_outputs[i]), self._normalize_tensor(pred_gt_outputs[i].detach()))
            perception_loss += layer_loss
            layer_losses.append((i, layer_loss.item()))

        
        # compute the image similarity loss
        img_loss = self.image_loss(output, target, mask=mask)

        # log the current losses
        self.cur_seg_loss = perception_loss.detach().cpu().numpy()
        self.cur_img_loss = img_loss.detach().cpu().numpy()

        return perception_loss * (1 - self.image_loss_weight) + img_loss * self.image_loss_weight

    def _normalize_tensor(self, in_feat,eps=1e-10):
        norm_factor = torch.sqrt(torch.sum(in_feat**2,dim=1,keepdim=True))
        return in_feat/(norm_factor+eps)
    

if __name__ == "__main__":
    # Example usage
    region = "AB"  # or "HN", "TH"
    syn_perception_loss = MaskedAnatomicalPerceptionLoss(dataset_name_seg=800, image_loss_weight=0.5, perception_masked=True)

    # # Dummy data for testing
    output = torch.randn(2, 1, 40, 192, 192)  # Example output from a model
    target = torch.randn(2, 1, 40, 192, 192)  # Example ground truth mask
    mask = torch.randint(0, 2, (2, 1, 40, 192, 192))  # Example mask
    output = output.to(device='cuda', dtype=torch.float16)  # Move to GPU if available
    target = target.to(device='cuda', dtype=torch.float16)  # Move to GPU if available
    mask = mask.to(device='cuda', dtype=torch.float16)  # Move to GPU if available

    loss_value = syn_perception_loss(output, target, mask)
    print(loss_value)
    print(f"Loss value: {loss_value.item()}")