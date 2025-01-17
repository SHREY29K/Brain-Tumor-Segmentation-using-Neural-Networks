# Brain Tumor Segmentation (BTS) using UNet

This project provides an in-depth approach to brain tumor segmentation from MRI images using the UNet model architecture. The implementation is developed in PyTorch and utilizes image segmentation techniques to identify tumor regions within MRI scans. A comprehensive training, validation, and testing pipeline is implemented to evaluate model performance with metrics such as Dice coefficient, Intersection over Union (IoU), and Binary Cross-Entropy loss combined with Dice loss.

## Table of Contents
1. **Introduction**
2. **Dataset**
3. **Model Architecture**
4. **Training and Evaluation**
5. **Results**
6. **Usage Instructions**

## Introduction
This project leverages deep learning for automatic brain tumor segmentation on MRI images. The UNet architecture was chosen for its effective feature extraction and spatial reconstruction capabilities, making it well-suited for medical image segmentation.

## Dataset
The dataset, downloaded from [Kaggle](https://www.kaggle.com/datasets/mateuszbuda/lgg-mri-segmentation), contains MRI scans with corresponding mask images that label tumor regions. The dataset is preprocessed and split into training, validation, and test sets.

## Model Architecture
The UNet model architecture consists of an encoder-decoder structure with skip connections that help retain spatial information. Convolutional layers in the encoder extract features, and upsampling in the decoder reconstructs the segmented regions.

## Training and Evaluation
The training process optimizes the combined Binary Cross-Entropy and Dice loss function to handle class imbalance. Performance metrics include:
- **Dice Coefficient**: Measures overlap between prediction and ground truth.
- **IoU (Intersection over Union)**: Evaluates accuracy of the segmentation.
  
Training is performed over 30 epochs with an Adam optimizer and a learning rate scheduler to enhance learning stability.
