# Welcome to nnsyn! ( 🏆 1st place in MICCAI-SynthRAD2025 challenge)
This repo holds the code and docker, which won 1st place in MR-to-CT synthesis task in MICCAI SynthRAD2025 challenge. 
- [SynthRAD2025 Task1 Final Leaderboard](https://synthrad2025.grand-challenge.org/evaluation/test-task-1-mri/leaderboard/) - 🥇 1st place
- [SynthRAD2025 Task2 Post-challenge Leaderboard](https://synthrad2025.grand-challenge.org/evaluation/post-challenge-task-2-cbct/leaderboard/)

# ✨ What is nnsyn? Self-configured framework for medical image synthesis
In this project, we would like to produce a user-friendly, mask-supported, extendable framework for medical image synthesis. We incorporated new CT preprocessing, new network architecture, new loss functions, and new evaluation metrics for image synthesis tasks. 

# 🌟 Feature highlights:
- [x] oneliner preprocessing
- [x] oneliner training (support masked loss, support MedNext)
- [x] oneliner inference
- [x] train a dedicated segmentation branch for perception loss
- [x] optional advanced experiment tracking with AIM
<!-- - [ ] oneliner evaluation -->


# 🚀 Installation:
```bash
git clone git@github.com:bowenxin/nnsyn.git
cd nnsyn
pip install -e .
```

# 📄 Quick start


First, export environment variables :
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
|           |-- PATIENT_0001.mha
|       |-- TARGET_IMAGES
|           |-- PATIENT_0001.mha
|       |-- MASKS (optional)
|           |-- PATIENT.mha (optional)
|       |-- LABELS (optional)
|           |-- PATIENT.mha (optional)
|-- nnUNet_raw
|   |-- DatasetXXX_YYY
|-- nnUNet_preprocessed
|   |-- DatasetXXX_YYY
|-- nnUNet_results
|   |-- DatasetXXX_YYY
```

Plan experiments and preprocess : 
```bash
nnsyn_plan_and_preprocess -d 960 -c 3d_fullres -pl nnUNetPlannerResEncL -p nnUNetResEncUNetLPlans  --preprocessing_input MR --preprocessing_target CT \
--data_origin_path 'PATH_TO/ORIGIN/synthrad2025_task1_mri2ct_AB'
```

Prepare dataset and preprocess for the segmentation branch :
```bash
nnsyn_plan_and_preprocess_seg -d 961 -ds 960 -c 3d_fullres -p nnUNetResEncUNetLPlans --data_origin_path 'PATH_TO/ORIGIN/synthrad2025_task1_mri2ct_AB' --preprocessing_target CT
```

Train the segmentation branch for perception loss :
```bash
git swtich nnunetv2
nnUNetv2_train 961 3d_fullres 0 -tr nnUNetTrainer -p nnUNetResEncUNetLPlans_Dataset960 --c
``` 

Train the synthesis network : 
```bash
git swtich main
nnsyn_train 960 3d_fullres 0 -tr nnUNetTrainer_nnsyn_loss_masked_perception_masked_track -p nnUNetResEncUNetLPlans
```

Inference :
```bash
nnsyn_predict -d DatasetY -i INPUT -o OUTPUT -m MASK -c 3d_fullres -p nnUNetPlans -tr nnUNetTrainerMRCT -f FOLD
```

<!-- # Citation

Along with the original nnUNet paper :

    Isensee, F., Jaeger, P. F., Kohl, S. A., Petersen, J., & Maier-Hein, K. H. (2021). nnU-Net: a self-configuring 
    method for deep learning-based biomedical image segmentation. Nature methods, 18(2), 203-211. -->
    
# 🤝 Credit
This project was build upon nnUNet_translation, nnUNet-v2, and TriALS. All awesome stuff. Please do not hesitate to check them out. 
- nnUNet_translateion: https://github.com/Phyrise/nnUNet_translation
- nnUNet-v2: https://github.com/MIC-DKFZ/nnUNet
- TriALS: https://github.com/xmed-lab/TriALS

# 📜 License

# Badges
