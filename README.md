# Welcome to nnsyn! 
This repo holds the code and docker winning 1st in MR-to-CT synthesis task in MICCAI SynthRAD2025 challenge. 

# 🚀 What is nnsyn? Self-configured deep learning framework for medical image synthesis
In this project, we would like create an easy-to-use, but extendable workflow for medical image synthesis task, similar to nnUNet-v2. We incorporated different preprocessing, new architectures, new evaluation metrics for image synthesis tasks. 

# 🚀 Feature highlights:
- [x] Easy-to-configure preprocessing
- Extendable beyond nnUNet
- Advanced experiemnt tracking - AIM
- [x] Support mask 
- Embedded synthesis metrics

# 🚀 Installation:
```bash
# I recommend creating a dedicated environment
git clone https://github.com/Phyrise/nnUNet_translation 
cd nnUNet_translation
pip install -e .
```
The `pip install` command should install the modified [batchgenerators](https://github.com/Phyrise/batchgenerators_translation) and [dynamic-network-architectures](https://github.com/Phyrise/dynamic-network-architectures_translation) repos.

# 🚀 Quick start


First, export variables :
```bash
export nnUNet_raw="path_to/nnUNet_raw"
export nnUNet_preprocessed="path_to/nnUNet_preprocessed"
export nnUNet_results="path_to/nnUNet_results"
```

Organise your data into ```"PATH_TO/ORIGIN"```:
```bash
DATA_STRUCT:
|-- ORIGIN
|   |-- Dataset_MRI2CT
|       |-- INPUT_IMAGES
|       |   |-- PATIENT_0001.mha
|       |-- TARGET_IMAGES
|           |-- PATIENT_0001.mha
|       |-- MASKS (optional)
|           |-- PATIENT_0001.mha (optional)
|-- nnUNet_raw
|   |-- DatasetXXX_YYY
|-- nnUNet_preprocessed
|   |-- DatasetXXX_YYY
|-- nnUNet_results
|   |-- DatasetXXX_YYY
```

Plan experiments and preprocess : 
```bash
nnsyn_plan_and_preprocess -d 982 -c 3d_fullres -pl nnUNetPlannerResEncL -p nnUNetResEncUNetLPlans  --preprocessing_input MR --preprocessing_target CT \
--data_origin_path 'PATH_TO/ORIGIN/Synthrad2025_MRI2CT_AB'
```

Train the network : 
```bash
nnUNetv2_train 290 3d_fullres 0 -tr nnUNetTrainerMRCT_track -p nnUNetResEncUNetLPlans
```

inference :
```bash
 nnUNetv2_predict -d DatasetY -i INPUT -o OUTPUT -c 3d_fullres -p nnUNetPlans -tr nnUNetTrainerMRCT -f FOLD [optional : -chk checkpoint_best.pth -step_size 0.5 --rec (mean,median)]
```

# Citation

Along with the original nnUNet paper :

    Isensee, F., Jaeger, P. F., Kohl, S. A., Petersen, J., & Maier-Hein, K. H. (2021). nnU-Net: a self-configuring 
    method for deep learning-based biomedical image segmentation. Nature methods, 18(2), 203-211.

# 🚀 Tested use cases : 
- MRI-to-CT translation: lung, abdomen, head-and-neck
- CBCT-to-CT translation: lung, abdomen, head-and-neck
    
# 🚀 Credit
This project was build upon nnUNet_translation, nnUNet-v2, and TriALS. All awesome stuff. Please do not hesitate to check them out. 
- nnUNet_translateion: https://github.com/Phyrise/nnUNet_translation
- nnUNet-v2: https://github.com/MIC-DKFZ/nnUNet
- TriALS: https://github.com/xmed-lab/TriALS

# 🚀 License

# 🚀 Badges
