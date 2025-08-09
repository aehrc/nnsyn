module load miniconda3
module load cuda/11.8.0
conda create --prefix /datasets/work/hb-synthrad2023/work/synthrad2025/bw_workplace/envs/nnunet_trans2 python==3.11
source activate /datasets/work/hb-synthrad2023/work/synthrad2025/bw_workplace/envs/nnunet_trans2

git clone /datasets/work/hb-synthrad2023/work/synthrad2025/bw_workplace/ref/nnUNet_translation
cd nnUNet_translation
pip install -e .

export nnUNet_raw="/datasets/work/hb-synthrad2023/work/synthrad2025/bw_workplace/data/nnunet_struct/raw"
export nnUNet_preprocessed="/datasets/work/hb-synthrad2023/work/synthrad2025/bw_workplace/data/nnunet_struct/preprocessed"
export nnUNet_results="/datasets/work/hb-synthrad2023/work/synthrad2025/bw_workplace/data/nnunet_struct/results"


# dependencies
pip install aim
pip install 'monai[all]' 
pip install 'monai[itk]'
pip install hiddenlayer

# aim commands
aim server --host virga.hpc.csiro.au --port 53800 --repo /datasets/work/hb-synthrad2023/work/synthrad2025/bw_workplace/output/runs
aim up --host 127.0.0.1 --port 43800 --repo /datasets/work/hb-synthrad2023/work/synthrad2025/bw_workplace/output/runs --workers=4
aim up --host 127.0.0.1 --port 43800 --repo /datasets/work/hb-synthrad2023/work/synthrad2025/bw_workplace/data/nnunet_struct/results/runs_aim --workers=4
# in the browser, go to http://localhost:43800/


conda create --prefix /datasets/work/hb-synthrad2023/work/synthrad2025/bw_workplace/envs/simple_elastix python==3.13
source activate /datasets/work/hb-synthrad2023/work/synthrad2025/bw_workplace/envs/simple_elastix
pip install SimpleITK-SimpleElastix==2.5.0.dev49 # needs python version>=12

conda create --prefix /datasets/work/hb-synthrad2023/work/synthrad2025/bw_workplace/envs/nnunet
source activate /datasets/work/hb-synthrad2023/work/synthrad2025/bw_workplace/envs/nnunet
module load cuda/11.8.0
pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu118
