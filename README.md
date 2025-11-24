[![Static Badge](https://img.shields.io/badge/MICCAI_Challenge-SynthRAD2025-blue)](https://synthrad2025.grand-challenge.org/participants/registration/create/)
[![Static Badge](https://img.shields.io/badge/huggingface-docker-orange)](https://huggingface.co/aehrc/synthrad2025_docker)
[![Static Badge](https://img.shields.io/badge/huggingface-demo-orange)](https://huggingface.co/spaces/aehrc/Synthrad2025)




# Welcome to nnsyn! ( 🏆 1st place in MICCAI-SynthRAD2025 MR-CT synthesis challenge)
This repo holds the code and docker, which won 1st place in MR-to-CT synthesis task in MICCAI SynthRAD2025 challenge. 
- [SynthRAD2025 Task1 Final Leaderboard](https://synthrad2025.grand-challenge.org/evaluation/test-task-1-mri/leaderboard/) - 🥇 1st place [15-Aug-2025]
- [SynthRAD2025 Task2 Post-challenge Leaderboard](https://synthrad2025.grand-challenge.org/evaluation/post-challenge-task-2-cbct/leaderboard/) - 🥇 1st place [post-challenge, 17-Aug-2025]

# ✨ What is nnsyn? Self-configured framework for medical image synthesis
In this project, we would like to produce a user-friendly, mask-supported, extendable framework for medical image synthesis. We incorporated new CT preprocessing, new network architectures, new loss functions, and new evaluation metrics for image synthesis tasks. 

# 🌟 Feature highlights:
- [x] Oneliner preprocessing
- [x] Oneliner training (support masked loss, support MedNext)
- [x] Oneliner inference
- [x] Train a dedicated segmentation branch for perception loss
- [x] Optional advanced experiment tracking with AIM
- [ ] Support on multimodal imaging inputs (ongoing)
<!-- - [ ] oneliner evaluation -->


# 🚀 Installation:
```bash
git clone git@github.com:aehrc/nnsyn.git
cd nnsyn
pip install -e .
```

# 📄 Quick start


First, export environment variables :
```bash
export nnsyn_origin_dataset = "path_to/nnsyn_origin/synthrad2025_task1_mri2ct_AB"
export nnUNet_raw="path_to/nnUNet_raw"
export nnUNet_preprocessed="path_to/nnUNet_preprocessed"
export nnUNet_results="path_to/nnUNet_results"
```

Organise your data into ```"nnsyn_origin_dataset"```. The "MASKS" folder contains the body contour, while the 'LABELS' folder contains segmentation labels. An example of dataset.json in [example](documentation/dataset_format.md). Currently, data needs to be convert to .mha. 
```bash
DATA_STRUCT:
|-- nnsyn_origin
|   |-- synthrad2025_task1_mri2ct_AB
|       |-- INPUT_IMAGES
|           |-- PATIENT_1_0001.mha
|       |-- TARGET_IMAGES
|           |-- PATIENT_1_0001.mha
|       |-- MASKS (optional)
|           |-- PATIENT_1.mha
|       |-- LABELS (optional)
|           |-- PATIENT_1.mha
|           |-- dataset.json 
|-- nnUNet_raw
|   |-- DatasetXXX_YYY
|-- nnUNet_preprocessed
|   |-- DatasetXXX_YYY
|-- nnUNet_results
|   |-- DatasetXXX_YYY
```

Plan experiments and preprocess for the synthesis model. 
```bash
nnsyn_plan_and_preprocess -d 960 -c 3d_fullres -pl nnUNetPlannerResEncL -p nnUNetResEncUNetLPlans  --preprocessing_input MR --preprocessing_target CT 
```

(For loss_map) Prepare dataset and preprocess for the segmentation model. The plan will be transfered from synthesis model (960) to segmentation model (961). 
```bash
nnsyn_plan_and_preprocess_seg -d 960 -dseg 961 -c 3d_fullres -p nnUNetResEncUNetLPlans
```

(For loss_map) Train the segmentation model for perception loss. We first switch to github segmentation branch (nnunetv2), train the segmentation model, and then switch back to the github synthesis branch (main). 
```bash
git switch nnunetv2
nnUNetv2_train 961 3d_fullres 0 -tr nnUNetTrainer -p nnUNetResEncUNetLPlans_Dataset960 --c
git switch main
``` 

Train the synthesis network with Masked Anatomical Perception (map) loss: 
```bash
nnsyn_train 960 3d_fullres 0 -tr nnUNetTrainer_nnsyn_loss_map -p nnUNetResEncUNetLPlans
```

Inference :
```bash
nnsyn_predict -d 960 -i INPUT_PATH -o OUTPUT_PATH -m MASK_PATH -c 3d_fullres -p nnUNetResEncUNetLPlans -tr nnUNetTrainer_nnsyn_loss_map -f 0
```

<!-- # Citation

Along with the original nnUNet paper :

    Isensee, F., Jaeger, P. F., Kohl, S. A., Petersen, J., & Maier-Hein, K. H. (2021). nnU-Net: a self-configuring 
    method for deep learning-based biomedical image segmentation. Nature methods, 18(2), 203-211. -->
    
# 🤝 Credit
This project was build upon nnUNet_translation, nnUNet-v2, and TriALS. All awesome stuff. Please do not hesitate to check them out. 
- nnUNet_translation: https://github.com/Phyrise/nnUNet_translation - [Paper1](https://doi.org/10.1007/978-3-031-73281-2_3) - [Paper2](https://iopscience.iop.org/article/10.1088/1361-6560/adea07)
- nnUNet-v2: https://github.com/MIC-DKFZ/nnUNet - [Paper](https://doi.org/10.1038/s41592-020-01008-z)
- TriALS: https://github.com/xmed-lab/TriALS

# ℹ️ Docker & huggingface space
Please find the instructions to reproduce the docker image for SynthRAD2025 task1 and task2 at https://huggingface.co/aehrc/synthrad2025_docker. 

Also, we provided a demo at huggingface space. Because only cpu resources are available for demo, it would be a bit slow (5 min/volume). On gpu, the inference time is 9 seconds/volume. 
