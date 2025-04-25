
# 🧠 Brain Tumor Segmentation using Deep Learning (UNet)

This repository contains a deep learning project focused on **brain tumor segmentation from MRI scans** using advanced image segmentation techniques. The model is built using **PyTorch**, primarily leveraging the **U-Net architecture**, and trained on the **LGG MRI Segmentation dataset** from Kaggle.

> 📌 A research paper based on this work is currently in development.

---

## 🔬 Project Overview

Brain tumor segmentation is crucial in medical diagnosis and treatment planning. This project aims to develop an accurate segmentation model using **convolutional neural networks (CNNs)** with **U-Net**, a well-known architecture for biomedical image segmentation.

---

## 📁 Dataset

- **Name**: LGG MRI Segmentation Dataset  
- **Source**: [Kaggle - mateuszbuda/lgg-mri-segmentation](https://www.kaggle.com/datasets/mateuszbuda/lgg-mri-segmentation)
- **Description**: This dataset contains MRI scan images of patients with **Lower Grade Glioma (LGG)** tumors, including corresponding ground truth segmentation masks.

---

## ⚙️ Tools & Technologies

| Area                 | Tools/Frameworks                                    |
|----------------------|-----------------------------------------------------|
| Programming Language | Python                                              |
| Deep Learning        | PyTorch, Torchvision                                |
| Image Augmentation   | Albumentations                                      |
| Data Handling        | NumPy, Pandas                                       |
| Visualization        | Matplotlib, OpenCV, PIL, ImageGrid                  |
| Training Utilities   | tqdm, sklearn (train/test split)                    |

---

## 🧠 Model Architecture

### UNet

UNet is a symmetric CNN architecture known for its effectiveness in medical image segmentation. It consists of:

- **Encoder (Contracting path)**: Extracts features via Conv → ReLU → MaxPool layers.
- **Decoder (Expanding path)**: Performs upsampling, concatenation, and convolution.
- **Skip Connections**: Help retain spatial information.

---

## 🛠️ Preprocessing & Augmentation

- Normalization of images
- Resizing to a fixed dimension
- Data Augmentation:
  - Horizontal and vertical flips
  - Rotation, shift, scale, and brightness adjustments (via Albumentations)

---

## 🧪 Training Details

- **Loss Function**: Dice Loss (to maximize overlap between predicted and true mask)
- **Optimizer**: Adam
- **Learning Rate**: 1e-4 (tunable)
- **Batch Size**: 8
- **Epochs**: Configurable
- **Device**: CUDA (if available), else CPU

---

## 📊 Evaluation Metrics

- **Dice Coefficient**
- **IoU (Intersection over Union)**
- **Pixel-wise Accuracy**
- **Visual comparison** between ground truth and predicted masks

---

## 📈 Results

- The model achieves **high segmentation accuracy** on validation images.
- Demonstrates effective tumor boundary detection.

---

## 🧠 Sample Results

![Example](sample_prediction_visualization.png) <!-- Add if you have visualizations saved -->

---

## 📌 How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/brain-tumor-segmentation.git
   cd brain-tumor-segmentation
   ```

2. Install requirements:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the notebook:
   ```bash
   jupyter notebook Brain_Tumor_Segmentation_(NN).ipynb
   ```

4. (Optional) Download dataset from Kaggle and place in the correct path.

---

## 🧪 Research Paper in Progress

We are currently working on a detailed **research publication** based on this project. It will include:

- Deeper model comparisons (UNet vs ResUNet, etc.)
- Dataset exploration
- Experimental results with various loss functions
- Performance on augmented data
- Future directions for improved diagnostic assistance

---

## 🤝 Contributing

Feel free to fork, raise issues, or submit pull requests!

---

## 📄 License

This project is licensed under the **MIT License**.
