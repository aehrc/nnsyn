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
from nnunetv2.utilities.get_network_from_plans import get_network_from_plans_and_class
from nnunetv2.training.loss.unet import ResidualEncoderUNet
from nnunetv2.training.loss.ssim_losses import SSIMLoss


# https://github.com/richzhang/PerceptualSimilarity/blob/master/lpips/lpips.py
# https://github.com/Project-MONAI/GenerativeModels/blob/main/generative/losses/perceptual.py


class SynPerceptionLoss(nn.Module):
    def __init__(self, region: str, image_loss_weight: float = 0.5, perception_masked=False):
        """
        Initializes the SynSegLoss module.
        region can be "AB", "HN", or "TH" to specify the anatomical region.
        """
        super(SynPerceptionLoss, self).__init__()
        self.region = region

        # load the trained segmentor model
        self.seg_model, self.seg_model_info = self._load_trained_segmentor(region)
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
    
    
    
    def _load_trained_segmentor(self, region: str):
        segmentor_training_output_dir = {
            "AB": "/datasets/work/hb-synthrad2023/work/synthrad2025/bw_workplace/data/nnunet_struct/results/Dataset800_SEGMENTATION_synthrad2025_task1_CT_AB_aligned_to_Dataset261/nnUNetTrainer__nnUNetResEncUNetLPlans_Dataset261__3d_fullres",
            "HN": "/datasets/work/hb-synthrad2023/work/synthrad2025/bw_workplace/data/nnunet_struct/results/Dataset801_SEGMENTATION_synthrad2025_task1_CT_HN_aligned_to_Dataset263/nnUNetTrainer__nnUNetResEncUNetLPlans_Dataset263__3d_fullres",
            "TH": "/datasets/work/hb-synthrad2023/work/synthrad2025/bw_workplace/data/nnunet_struct/results/Dataset802_SEGMENTATION_synthrad2025_task1_CT_TH_aligned_to_Dataset265/nnUNetTrainer__nnUNetResEncUNetLPlans_Dataset265__3d_fullres"
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


class SynPerceptionLoss_L2(SynPerceptionLoss):
    def __init__(self, region: str, image_loss_weight: float = 0.5):
        super(SynPerceptionLoss_L2, self).__init__(region, image_loss_weight)

    def forward(self, output: torch.Tensor, target: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        # compute the segmentation loss
        pred_outputs = self.seg_model(output)
        pred_gt_outputs = self.seg_model(target)

        diffs = {}
        res = []
        perception_loss = 0

        for i in range(self.seg_model_info['n_stages']):
            diffs[i] = (self._normalize_tensor(pred_outputs[i]) - self._normalize_tensor(pred_gt_outputs[i].detach())) ** 2
            x = diffs[i].sum(dim=1,keepdim=True)
            res.append(self._spatial_average_3d(x, keepdim=True))
            perception_loss += res[i]

        # compute the image similarity loss
        img_loss = self.image_loss(output, target, mask=mask)
        # print("Perception Loss:", perception_loss, perception_loss.shape)
        # print("Image Loss:", img_loss, img_loss.shape)


        # log the current losses
        self.cur_seg_loss = perception_loss.detach().cpu().numpy()
        self.cur_img_loss = img_loss.detach().cpu().numpy()

        return perception_loss.mean() * (1 - self.image_loss_weight) + img_loss * self.image_loss_weight
    
    def _spatial_average_3d(self, x: torch.Tensor, keepdim: bool = True) -> torch.Tensor:
        return x.mean([2, 3, 4], keepdim=keepdim)
    

class SynPerceptionLoss_L2_ssim(SynPerceptionLoss):
    def __init__(self, region: str, image_loss_weight: float = 0.5):
        super(SynPerceptionLoss_L2_ssim, self).__init__(region, image_loss_weight)
        self.ssim_loss = SSIMLoss(spatial_dims=3, data_range=1.0, kernel_type='gaussian', win_size=11, kernel_sigma=1.5, k1=0.01, k2=0.03)

    def forward(self, output: torch.Tensor, target: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        # compute the segmentation loss
        pred_outputs = self.seg_model(output)
        pred_gt_outputs = self.seg_model(target)

        diffs = {}
        res = []
        perception_loss = 0

        for i in range(self.seg_model_info['n_stages']):
            diffs[i] = (self._normalize_tensor(pred_outputs[i]) - self._normalize_tensor(pred_gt_outputs[i].detach())) ** 2
            x = diffs[i].sum(dim=1,keepdim=True)
            res.append(self._spatial_average_3d(x, keepdim=True))
            perception_loss += res[i]

        # compute the image similarity loss
        img_loss = self.image_loss(output, target, mask=mask)
        img_loss += self.ssim_loss(output, target, mask=mask)
        # print("Perception Loss:", perception_loss, perception_loss.shape)
        # print("Image Loss:", img_loss, img_loss.shape)


        # log the current losses
        self.cur_seg_loss = perception_loss.detach().cpu().numpy()
        self.cur_img_loss = img_loss.detach().cpu().numpy()

        return perception_loss.mean() * (1 - self.image_loss_weight) + img_loss * self.image_loss_weight
    
    def _spatial_average_3d(self, x: torch.Tensor, keepdim: bool = True) -> torch.Tensor:
        return x.mean([2, 3, 4], keepdim=keepdim)

if __name__ == "__main__":
    # Example usage
    region = "AB"  # or "HN", "TH"
    syn_perception_loss = SynPerceptionLoss_L2_ssim(region, image_loss_weight=0.5)
    
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