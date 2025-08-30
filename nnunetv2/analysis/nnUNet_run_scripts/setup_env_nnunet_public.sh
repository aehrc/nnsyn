module load miniconda3
module load cuda/11.8.0
conda create --prefix /datasets/work/hb-synthrad2023/work/synthrad2025/bw_workplace/envs/nnsyn_public python==3.11
source activate /datasets/work/hb-synthrad2023/work/synthrad2025/bw_workplace/envs/nnsyn_public

cd /datasets/work/hb-synthrad2023/work/synthrad2025/bw_workplace/ref/nnsyn
pip install -e .

export nnUNet_raw="/datasets/work/hb-synthrad2023/work/synthrad2025/bw_workplace/data/nnunet_struct/raw"
export nnUNet_preprocessed="/datasets/work/hb-synthrad2023/work/synthrad2025/bw_workplace/data/nnunet_struct/preprocessed"
export nnUNet_results="/datasets/work/hb-synthrad2023/work/synthrad2025/bw_workplace/data/nnunet_struct/results"