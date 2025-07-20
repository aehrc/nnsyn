from concurrent.futures import ThreadPoolExecutor
import os
import shutil
from tqdm import tqdm
import json


def process_file(data_path, dataset_path, modality_suffix="_0000"):
    os.makedirs(dataset_path, exist_ok=True)
    filename = data_path.split(os.sep)[-2]
    if not filename.endswith(f'{modality_suffix}.mha'):
        filename = filename + f'{modality_suffix}.mha'
    shutil.copy(data_path, os.path.join(dataset_path, filename))

train_test_split_fold0 = "/datasets/work/hb-synthrad2023/work/synthrad2025/bw_workplace/output/task1_AB/p2p3D/exp0_data0_128_128_128_aug3d_batchsize3_lr0.0002_cpu4/train_test_split_fold0.json"
split = json.load(open(train_test_split_fold0))
list_data_mri = split['images_test']
list_data_ct = [os.path.join(os.path.dirname(image), 'ct.mha') for image in list_data_mri]
list_data_mask = [os.path.join(os.path.dirname(image), 'mask.mha') for image in list_data_mri]
output_dir = "/datasets/work/hb-synthrad2023/work/synthrad2025/bw_workplace/data/nnunet_struct/raw/cross_val_5folds/fold0"

# with ThreadPoolExecutor() as executor:
#     list(tqdm(executor.map(lambda data_path: process_file(data_path, os.path.join(output_dir, 'image'), "_0000"), list_data_mri), total=len(list_data_mri)))

with ThreadPoolExecutor() as executor:
    list(tqdm(executor.map(lambda target_path: process_file(target_path, os.path.join(output_dir, 'gt_segmentations'), ""), list_data_ct), total=len(list_data_ct)))

with ThreadPoolExecutor() as executor:
    list(tqdm(executor.map(lambda target_path: process_file(target_path, os.path.join(output_dir, 'masks'), ""), list_data_mask), total=len(list_data_mask)))
