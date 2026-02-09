# Comparative Analysis of Convolutional and Vision Transformer Models for Automated Leukocyte Classification Enhanced by Generative Color Augmentation

![Python](https://img.shields.io/badge/python-3.10-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Status](https://img.shields.io/badge/Status-Completed-success)

## 📄 Abstract

The differential leukocyte count is a fundamental procedure in hematological diagnosis. However, manual analysis is time-consuming, subjective, and prone to inter-observer variability. This project presents a rigorous comparative study between **Convolutional Neural Networks (YOLOv11)** and **Vision Transformers (ViT)** for the classification of **14 distinct types** of leukocytes and artifacts.

Furthermore, we investigated the impact of **HistAuGAN** (Generative Adversarial Networks) for stain color augmentation. The results demonstrated the consistent superiority of the Vision Transformer architecture. The combination of **ViT-Base + HistAuGAN** achieved state-of-the-art performance with a **Macro F1-Score of 98.36%** and an overall accuracy of **99.75%**.

## 🧬 Dataset

The dataset used in this study consists of high-resolution images of peripheral blood smears collected at the Clinical Laboratory of the University of Vale do Itajaí (UNIVALI).

* [cite_start]**Total Samples:** 5,011 annotated cells[cite: 873].
* [cite_start]**Classes:** 14 categories, including mature cells, immature granulocytes (blasts, promyelocytes, myelocytes, metamyelocytes), and artifacts[cite: 899].
* [cite_start]**Data Split:** 80% Train, 10% Validation, 10% Test[cite: 920].

**Note:** Due to privacy and ethical restrictions, the raw images are hosted externally.
* 📥 **Download Dataset:** [Link to Zenodo/Kaggle] *(Insert your link here)*

### Class Distribution
| Class | Samples | Representative % |
| :--- | :---: | :---: |
| Segmented Neutrophil | 1752 | 34.96% |
| Lymphocyte | 1033 | 20.61% |
| Blast | 438 | 8.74% |
| ... | ... | ... |
| **Total** | **5011** | **100%** |

## 🛠️ Methodology & Architectures

We evaluated two distinct deep learning paradigms:

1.  **YOLOv11 (Ultralytics):** Tested in *Nano*, *Small*, and *Medium* variants. Adapted for classification tasks.
2.  [cite_start]**Vision Transformer (ViT):** Tested in *Small* and *Base* variants (patch size 16x16), utilizing self-attention mechanisms to capture global morphological features[cite: 935].

### Data Augmentation Strategy
To address laboratory staining variability, we employed **HistAuGAN**, a style-transfer GAN that decouples image content from style. [cite_start]This expanded the training set from 9,320 images to **55,920 images**, significantly improving generalization[cite: 947].

## 📊 Results

The models were evaluated on an independent test set. The metrics below represent the **Macro Average** to account for class imbalance.

### Impact of HistAuGAN (Performance Comparison)

| Architecture | Variant | Strategy | Precision (%) | Recall (%) | **F1-Score (%)** |
| :--- | :--- | :--- | :---: | :---: | :---: |
| **YOLOv11** | Medium | Standard | 91.75 | 91.64 | 91.70 |
| **YOLOv11** | Medium | **+ HistAuGAN** | 96.91 | 97.73 | 97.32 |
| **ViT** | Base | Standard | 98.02 | 97.45 | 97.74 |
| **ViT** | **Base** | **+ HistAuGAN** | **98.52** | **98.19** | **98.36** |

[cite_start]*[cite: 1137, 1142]*

### Confusion Matrix (ViT-Base + HistAuGAN)
[cite_start]The best model achieved near-perfect classification for mature cells and high robustness for immature stages (e.g., Blasts: 99% F1-Score)[cite: 1149].

*(Add your Confusion Matrix image here, e.g., `![Confusion Matrix](results/confusion_matrix.png)`)*

## 🚀 Installation & Usage

### Prerequisites
* Linux (Ubuntu 22.04 recommended) or Windows w/ WSL2
* Python 3.10+
* CUDA 11.8+ (for GPU acceleration)


### References
@inproceedings{HistAuGAN,
  author = {Wagner, S. J., Khalili, N., Sharma, R., Boxberg, M., Marr, C., de Back, W., Peng, T.},
  booktitle = {Medical Image Computing and Computer Assisted Intervention – MICCAI 2021},
  title = {Structure-Preserving Multi-Domain Stain Color Augmentation using Style-Transfer with Disentangled Representations},
  year = {2021}
}


### 1. Clone the repository
```bash
git clone [https://github.com/your-username/leukocyte-classification-vit-yolo.git](https://github.com/your-username/leukocyte-classification-vit-yolo.git)
cd leukocyte-classification-vit-yolo


