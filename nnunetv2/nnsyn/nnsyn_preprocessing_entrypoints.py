from nnunetv2.nnsyn.nnsyn_preprocessing import nnsyn_plan_and_preprocess
from nnunetv2.nnsyn.nnsyn_preprocessing_seg import nnsyn_plan_and_preprocess_seg

def nnsyn_plan_and_preprocess_entry():
    import argparse
    parser = argparse.ArgumentParser(description="nnUNet preprocessing for nnSyn")
    parser.add_argument('-d', '--dataset_id', type=int, required=True, help="Dataset ID (will create DatasetXXX_<name> and DatasetXXX+1_<name> in nnUNet_raw and nnUNet_preprocessed)")
    parser.add_argument('-c', '--configuration', type=str, default='3d_fullres', help="Configuration to preprocess (default: 3d_fullres)")
    parser.add_argument('-pl', '--planner_class', type=str, default='ExperimentPlanner', help="Experiment planner class (default: nnUNetExperimentPlanner3D_v2)")
    parser.add_argument('-p', '--plan', type=str, default='nnUNetPlans', help="Plan identifier (default: nnUNetPlans)")
    # parser.add_argument('--data_origin_path', type=str, required=True, help="Root path for nnUNet_raw and nnUNet_preprocessed (default: uses environment variables)")
    parser.add_argument('--preprocessing_input', type=str, required=True, help="Preprocessing for input data (e.g., CT, MR, synthrad, etc.)")
    parser.add_argument('--preprocessing_target', type=str, required=True, help="Preprocessing for target data (e.g., CT, MR, synthrad, etc.)")
    parser.add_argument('--dataset_name', type=str, default=None, required=False, help="Name for input dataset (e.g., CT, MR, synthrad, etc.)")
    parser.add_argument('--use_mask', action='store_true', default=False, help="Whether to use masks (default: False)")

    args = parser.parse_args()

    nnsyn_plan_and_preprocess(
    #    data_origin_path=args.data_origin_path,
       dataset_id=args.dataset_id,
       dataset_name=args.dataset_name,
       preprocessing_input=args.preprocessing_input,
       preprocessing_target=args.preprocessing_target,
       use_mask=args.use_mask,
       configuration=args.configuration,
       planner_class=args.planner_class,
       plan=args.plan,
   )
    
def nnsyn_plan_and_preprocess_seg_entry():
    import argparse
    parser = argparse.ArgumentParser(description="nnUNet preprocessing for nnSyn")
    parser.add_argument('-d', '--dataset_id_syn', type=int, required=True, help="Original Dataset ID (will create DatasetXXX_<name> and DatasetXXX+1_<name> in nnUNet_raw and nnUNet_preprocessed)")
    parser.add_argument('-dseg', '--dataset_id_seg', type=int, required=True, help="Segmentation Dataset ID (will create DatasetXXX_<name> and DatasetXXX+1_<name> in nnUNet_raw and nnUNet_preprocessed)")
    parser.add_argument('-c', '--configuration', type=str, default='3d_fullres', help="Configuration to preprocess (default: 3d_fullres)")
    parser.add_argument('-p', '--plan', type=str, default='nnUNetPlans', help="Plan identifier (default: nnUNetPlans)")
    # parser.add_argument('--data_origin_path', type=str, required=True, help="Root path for nnUNet_raw and nnUNet_preprocessed (default: uses environment variables)")
    parser.add_argument('--dataset_name', type=str, default=None, required=False, help="Name for input dataset (e.g., CT, MR, synthrad, etc.)")

    args = parser.parse_args()

    # nnsyn_plan_and_preprocess_seg(data_origin_path='/datasets/work/hb-synthrad2023/work/synthrad2025/bw_workplace/data/nnunet_struct/ORIGIN/synthrad2025_task1_mri2ct_AB', 
    #                           dataset_id_syn=961, dataset_id_src=960,
    #                           configuration='3d_fullres', plan='nnUNetResEncUNetLPlans')
    
    
    nnsyn_plan_and_preprocess_seg(
    #    data_origin_path=args.data_origin_path,
       dataset_id_syn=args.dataset_id_syn,
       dataset_id_seg=args.dataset_id_seg,
       dataset_name=args.dataset_name,
       configuration=args.configuration,
       plan=args.plan,
   )
    
    # example usage:
    # python -m nnsyn_plan_and_preprocess_entry -d 982 -c 3d_fullres -pl nnUNetPlannerResEncL -p nnUNetResEncUNetLPlans --data_origin_path '/datasets/work/hb-synthrad2023/work/synthrad2025/bw_workplace/data/nnunet_struct/ORIGIN/Synthrad2025_MRI2CT_AB' --preprocessing_input MR --preprocessing_target CT
    # nnsyn_plan_and_preprocess -d 960 -c 3d_fullres -pl nnUNetPlannerResEncL -p nnUNetResEncUNetLPlans --data_origin_path '/datasets/work/hb-synthrad2023/work/synthrad2025/bw_workplace/data/nnunet_struct/ORIGIN/Synthrad2025_MRI2CT_AB' --preprocessing_input MR --preprocessing_target CT
if __name__ == '__main__':
    nnsyn_plan_and_preprocess_entry()

    # example usage:
    # python -m nnsyn_plan_and_preprocess_seg_entry -d 961 -ds 960 -c 3d_fullres -p nnUNetResEncUNetLPlans --data_origin_path '/datasets/work/hb-synthrad2023/work/synthrad2025/bw_workplace/data/nnunet_struct/ORIGIN/synthrad2025_task1_mri2ct_AB'