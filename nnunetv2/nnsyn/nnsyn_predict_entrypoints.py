import os
import shutil
from nnunetv2.analysis.revert_normalisation import get_ct_normalisation_values, revert_normalisation
from nnunetv2.utilities.file_path_utilities import get_output_folder

def nnsyn_predict_entry():
    import argparse
    parser = argparse.ArgumentParser(description="nnUNet prediction for nnSyn")
    parser = argparse.ArgumentParser(description='Use this to run inference with nnU-Net. This function is used when '
                                                 'you want to manually specify a folder containing a trained nnU-Net '
                                                 'model. This is useful when the nnunet environment variables '
                                                 '(nnUNet_results) are not set.')
    parser.add_argument('-i', type=str, required=True,
                        help='input folder. Remember to use the correct channel numberings for your files (_0000 etc). '
                             'File endings must be the same as the training dataset!')
    parser.add_argument('-o', type=str, required=True,
                        help='Output folder. If it does not exist it will be created. Predicted segmentations will '
                             'have the same name as their source images.')
    parser.add_argument('-m', type=str, required=False, default=None,
                        help='input mask folder. If it does not exist it will be created. Predicted segmentations will '
                             'have the same name as their source images.')
    parser.add_argument('-d', type=str, required=True,
                        help='Dataset with which you would like to predict. You can specify either dataset name or id')
    parser.add_argument('-p', type=str, required=False, default='nnUNetPlans',
                        help='Plans identifier. Specify the plans in which the desired configuration is located. '
                             'Default: nnUNetPlans')
    parser.add_argument('-tr', type=str, required=False, default='nnUNetTrainer',
                        help='What nnU-Net trainer class was used for training? Default: nnUNetTrainer')
    parser.add_argument('-c', type=str, required=True,
                        help='nnU-Net configuration that should be used for prediction. Config must be located '
                             'in the plans specified with -p')
    parser.add_argument('-f', nargs='+', type=str, required=False, default=(0, 1, 2, 3, 4),
                        help='Specify the folds of the trained model that should be used for prediction. '
                             'Default: (0, 1, 2, 3, 4)')
    parser.add_argument('--revert_norm', action='store_true', required=False, default=False,
                        help='Set this flag if you want to undo the z-score normalization that is applied to '
                             'non-CT data. This will not give you the original intensities but will undo the '
                             'z-score normalization (mean and std are stored in the plans file). Note that this '
                             'only makes sense if you trained your model with revert_norm=True!')
    parser.add_argument('-step_size', type=float, required=False, default=0.5,
                        help='Step size for sliding window prediction. The larger it is the faster but less accurate '
                             'the prediction. Default: 0.5. Cannot be larger than 1. We recommend the default.')
    parser.add_argument('--disable_tta', action='store_true', required=False, default=False,
                        help='Set this flag to disable test time data augmentation in the form of mirroring. Faster, '
                             'but less accurate inference. Not recommended.')
    parser.add_argument('--verbose', action='store_true', help="Set this if you like being talked to. You will have "
                                                               "to be a good listener/reader.")
    parser.add_argument('--save_probabilities', action='store_true',
                        help='Set this to export predicted class "probabilities". Required if you want to ensemble '
                             'multiple configurations.')
    parser.add_argument('--continue_prediction', action='store_true',
                        help='Continue an aborted previous prediction (will not overwrite existing files)')
    parser.add_argument('-chk', type=str, required=False, default='checkpoint_final.pth',
                        help='Name of the checkpoint you want to use. Default: checkpoint_final.pth')
    parser.add_argument('-npp', type=int, required=False, default=3,
                        help='Number of processes used for preprocessing. More is not always better. Beware of '
                             'out-of-RAM issues. Default: 3')
    parser.add_argument('-nps', type=int, required=False, default=3,
                        help='Number of processes used for segmentation export. More is not always better. Beware of '
                             'out-of-RAM issues. Default: 3')
    parser.add_argument('-prev_stage_predictions', type=str, required=False, default=None,
                        help='Folder containing the predictions of the previous stage. Required for cascaded models.')
    parser.add_argument('-num_parts', type=int, required=False, default=1,
                        help='Number of separate nnUNetv2_predict call that you will be making. Default: 1 (= this one '
                             'call predicts everything)')
    parser.add_argument('-part_id', type=int, required=False, default=0,
                        help='If multiple nnUNetv2_predict exist, which one is this? IDs start with 0 can end with '
                             'num_parts - 1. So when you submit 5 nnUNetv2_predict calls you need to set -num_parts '
                             '5 and use -part_id 0, 1, 2, 3 and 4. Simple, right? Note: You are yourself responsible '
                             'to make these run on separate GPUs! Use CUDA_VISIBLE_DEVICES (google, yo!)')
    parser.add_argument('-device', type=str, default='cuda', required=False,
                        help="Use this to set the device the inference should run with. Available options are 'cuda' "
                             "(GPU), 'cpu' (CPU) and 'mps' (Apple M1/M2). Do NOT use this to set which GPU ID! "
                             "Use CUDA_VISIBLE_DEVICES=X nnUNetv2_predict [...] instead!")
    parser.add_argument('--disable_progress_bar', action='store_true', required=False, default=False,
                        help='Set this flag to disable progress bar. Recommended for HPC environments (non interactive '
                             'jobs)')
    parser.add_argument('--rec', type=str, default='mean', choices=['mean', 'median'],
                        help='Method of reconstruction: mean or median. Default is mean.')
    
    args = parser.parse_args()
    # change list args.f to a string
    args.f = ' '.join(args.f)

    os.system(f"nnUNetv2_predict -d {args.d} -i {args.i} -o {args.o} -c {args.c} -p {args.p} -tr {args.tr} -f {args.f} -chk {args.chk} ")

    # revert normalisation
    if args.revert_norm:
        print("Reverting normalisation...")

        model_folder = get_output_folder(args.d, args.tr, args.p, args.c)
        dataset_name = model_folder.split(os.sep)[-2]
        ct_plan_path = os.path.join(os.environ["nnUNet_preprocessed"], dataset_name, f"gt_{args.p}.json")
        if not os.path.exists(ct_plan_path):
            ct_plan_path = os.path.join(os.environ["nnUNet_preprocessed"], dataset_name, 'gt_plan', f"{args.p}.json")
        ct_mean, ct_std = get_ct_normalisation_values(ct_plan_path)
        revert_normalisation(args.o, ct_mean, ct_std, save_path=args.o + "_revert_norm", \
                         mask_path=args.m, mask_outside_value=-1000)
        # move previous results to backup
        if os.path.exists(args.o):
            shutil.move(args.o, args.o + "_revert_norm/backup_normalised")

if __name__ == '__main__':
    nnsyn_predict_entry()
    # nnsyn_predict -d 960 -i /datasets/work/hb-synthrad2023/work/synthrad2025/bw_workplace/data/nnunet_struct/raw/Dataset960_synthrad2025_task1_mri2ct_AB/imagesVal -o /datasets/work/hb-synthrad2023/work/synthrad2025/bw_workplace/data/nnunet_struct/raw/Dataset960_synthrad2025_task1_mri2ct_AB/imagesVal_results -m /datasets/work/hb-synthrad2023/work/synthrad2025/bw_workplace/data/nnunet_struct/raw/Dataset960_synthrad2025_task1_mri2ct_AB/imagesVal_mask -c 3d_fullres -p nnUNetResEncUNetLPlans -tr nnUNetTrainer_nnsyn_loss_masked_perception_masked_track -f 0 -chk checkpoint_best.pth --revert_norm