# Self-Supervised Learning for Stroke MRI Dataset

## Objective
Evaluate the effect of self-supervised pretraining on stroke lesion segmentation in MRI images.

## Method
Train the baseline model and other self-supervised models on ATLAS dataset. Evaluate downstream performance with ISLES'22 dataset. 
Code impmentation is based on https://github.com/Project-MONAI/tutorials/tree/main/self_supervised_pretraining#3-self-supervised-tasks

### Models
1. UNETR baseline
2. UNETR with weights transferred from pretrained ViT
