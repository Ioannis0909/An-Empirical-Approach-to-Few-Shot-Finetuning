# Few-Shot Learning with Prototypical Networks: A Comparative Study of CNN and Transformer Architectures

**Course**: 20600 - Deep Learning for Computer Vision
**Program**: MSc Data Science and Business Analytics
**Institution**: Università Bocconi
**Due Date**: 6th December, 2025

**Group Name**: Bolshevik Baddies 

**Group Members**:
- 3176145 - Ioannis Thomopoulos
- 3161719 - Jacopo Mattia D'Angelo
- 3325056 - Maximillian Rienth

**Instructor**: Prof. Chiara Plizzari

---

## Overview

This project implements and evaluates **Prototypical Networks** for few-shot learning tasks, comparing the performance of CNN-based (ResNet18) and Transformer-based (Vision Transformer B/16) backbone architectures. The study investigates different levels of model adaptation, from frozen feature extractors to fully fine-tuned models.

Few-shot learning addresses the challenge of training machine learning models with limited labeled examples per class. Prototypical Networks learn to classify new examples by computing distances to prototype representations of each class, making them particularly well-suited for this task.

---

## Reference Implementation

This project uses the **CloserLookFewShot** repo and implementation for a comparison to our own implementation (the folder: CloserLooKFewShot in our repo). It was slightly adapted so it would allow for the use of our pre-trained backbone (ResNet18_Weights.IMAGENET1K_V1), but the core methodology was left completely untouched. Please find their Paper, Code, and Citation below:

**Paper**: [A Closer Look at Few-shot Classification](https://arxiv.org/pdf/1904.04232)

**Repository**: [https://github.com/wyharveychen/CloserLookFewShot](https://github.com/wyharveychen/CloserLookFewShot)

**Citation**:
```bibtex
@inproceedings{chen2019closerfewshot,
  title={A Closer Look at Few-shot Classification},
  author={Chen, Wei-Yu and Liu, Yen-Cheng and Kira, Zsolt and Wang, Yu-Chiang Frank and Huang, Jia-Bin},
  booktitle={International Conference on Learning Representations},
  year={2019}
}
```
---

## Project Structure

```
00_Submission_Models/
├── Bird_Models/              # Models trained on CUB-200-2011 dataset
│   ├── CNN/
│   │   ├── Base_Model/          # Frozen ResNet18 baseline
│   │   ├── One_Layer/           # Frozen backbone + single projection layer
│   │   ├── Two_Layer/           # Frozen backbone + two-layer projection
│   │   └── Fully_Tuned/         # Fine-tuned ResNet18
│   ├── Transformers/
│   │   ├── Base_Model/          # Frozen ViT-B/16 baseline
│   │   ├── One_Layer/           # Frozen backbone + single projection layer
│   │   ├── Two_Layer/           # Frozen backbone + two-layer projection
│   │   └── Fully_Tuned/         # Fine-tuned ViT-B/16
│   └── run_all_models.sh        # Automated pipeline for Bird models
├── CAPTCHA_Models/           # Models trained on CAPTCHA dataset
│   ├── CNN/
│   │   ├── Base_Model/          # Frozen ResNet18 baseline
│   │   ├── One_Layer/           # Frozen backbone + single projection layer
│   │   ├── Two_Layer/           # Frozen backbone + two-layer projection
│   │   └── Fully_Tuned/         # Fine-tuned ResNet18
│   ├── Transformers/
│   │   ├── Base_Model/          # Frozen ViT-B/16 baseline
│   │   ├── One_Layer/           # Frozen backbone + single projection layer
│   │   ├── Two_Layer/           # Frozen backbone + two-layer projection
│   │   └── Fully_Tuned/         # Fine-tuned ViT-B/16
│   └── run_all_models.sh        # Automated pipeline for CAPTCHA models
├── CloserLookFewShot/        # Reference implementation (original paper)
├── analyze_embeddings_all.py  # Embedding space visualization
├── test_on_train_classes.py    # Training class evaluation script
└── README.md
```

Each model variant contains:
- `main.py` - Entry point and training/evaluation pipeline
- `MODEL.py` - Model architecture definition
- `PREP.py` - Data preprocessing and loading
- `FEW_SHOT.py` - Few-shot evaluation protocol

---

## Datasets

### CUB-200-2011 (Caltech-UCSD Birds)

A benchmark dataset for fine-grained visual categorization containing 11,788 images across 200 bird species. Used for training the models in `Bird_Models/`.

**Download**: [https://data.caltech.edu/records/65de6-vp158](https://data.caltech.edu/records/65de6-vp158)

**Structure**:
```
CUB_200_2011/
├── images/
│   ├── 001.Black_footed_Albatross/
│   ├── 002.Laysan_Albatross/
│   └── ...
└── classes.txt
```

### CAPTCHA Dataset

A dataset of CAPTCHA images for character recognition. Used for training the models in `CAPTCHA_Models/`.

**Download**: [https://www.kaggle.com/datasets/mikhailma/test-dataset?resource=download](https://www.kaggle.com/datasets/mikhailma/test-dataset?resource=download)

---

## Model Architectures

### Backbone Networks

**CNN-based Models (ResNet18)**
- Base Architecture: ResNet18 pretrained on ImageNet-1K
- Embedding Dimension: 512-D (base) or 256-D (with projection layers)

**Transformer-based Models (ViT-B/16)**
- Base Architecture: Vision Transformer Base with 16×16 patch size, pretrained on ImageNet-1K
- Embedding Dimension: 768-D (CLS token) or 512-D (after projection)

### Training Strategies

| Variant | Description | Computational Cost |
|---------|-------------|-------------------|
| **Base Model** | Frozen pretrained backbone, no additional layers | Low |
| **One Layer** | Frozen backbone + single linear projection layer | Low |
| **Two Layer** | Frozen backbone + two-layer MLP projection | Low-Medium |
| **Fully Tuned** | End-to-end fine-tuning of entire backbone | High |

---

## Installation

### Prerequisites

- Python 3.8+
- CUDA-enabled GPU (Will not run on Apple Sillicon)
- 16GB+ GPU memory (for Fully-Tuned variants)

### Required Libraries

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install timm numpy matplotlib seaborn scikit-learn tqdm pandas
```

Minimum versions:
- PyTorch >= 2.0.0
- torchvision >= 0.15.0
- timm >= 0.9.0 (for ViT models)

---

## Usage

### Quick Start

#### 1. Download Dataset

```bash
wget https://data.caltech.edu/records/65de6-vp158/files/CUB_200_2011.tgz
tar -xzf CUB_200_2011.tgz
```

#### 2. Run Bird Models (Automated)

```bash
cd Bird_Models
chmod +x run_all_models.sh
./run_all_models.sh CUB_200_2011
```

This script trains and evaluates 6 trainable model variants for bird classification:
- CNN: One-Layer, Two-Layer, Fully-Tuned
- Transformers: One-Layer, Two-Layer, Fully-Tuned

#### 3. Run CAPTCHA Models (Automated)

```bash
cd CAPTCHA_Models
chmod +x run_all_models.sh
./run_all_models.sh <CAPTCHA_DATASET_PATH>
```

This script trains and evaluates 6 trainable model variants for CAPTCHA classification.

**Note**: The bash scripts do NOT run the baseline models (CNN/Base_Model and Transformers/Base_Model). These must be run individually as shown below.

#### 4. Run Baseline Models (Manual)

```bash
# Bird - CNN Baseline
cd Bird_Models/CNN/Base_Model
python main.py \
    --data_root ../../../CUB_200_2011/images \
    --classes_file ../../../CUB_200_2011/classes.txt \
    --seed 99

# Bird - Transformer Baseline
cd Bird_Models/Transformers/Base_Model
python main.py \
    --data_root ../../../CUB_200_2011/images \
    --classes_file ../../../CUB_200_2011/classes.txt \
    --seed 99

# CAPTCHA - CNN Baseline
cd CAPTCHA_Models/CNN/Base_Model
python main.py \
    --data_root <CAPTCHA_DATASET_PATH> \
    --seed 99

# CAPTCHA - Transformer Baseline
cd CAPTCHA_Models/Transformers/Base_Model
python main.py \
    --data_root <CAPTCHA_DATASET_PATH> \
    --seed 99
```

### Running Individual Models

```bash
# Navigate to model directory (example: Bird Models)
cd Bird_Models/CNN/One_Layer

# Training + Evaluation (default)
python main.py \
    --data_root ../../../CUB_200_2011/images \
    --classes_file ../../../CUB_200_2011/classes.txt \
    --seed 99 \
    --mode both

# Training only
python main.py ... --mode train

# Evaluation only
python main.py ... --mode eval --checkpoint outputs/best_model.pt
```

```bash
# Navigate to model directory (example: CAPTCHA Models)
cd CAPTCHA_Models/CNN/One_Layer

# Training + Evaluation (default)
python main.py \
    --data_root <CAPTCHA_DATASET_PATH> \
    --seed 99 \
    --mode both
```

### Analysis Scripts

```bash
# Embedding space visualization (Bird Models)
python analyze_embeddings_all.py

# Test on training classes
python test_on_train_classes.py
```

---

## Requirements

### Hardware
- **Minimum**: NVIDIA GPU with 8GB VRAM
- **Recommended**: NVIDIA GPU with 16GB VRAM (for Fully-Tuned models)

### Software

PyTorch, torchvision, timm, NumPy, Matplotlib, Seaborn, scikit-learn, tqdm, pandas

---

**Group**: Bolshevik Baddies
**Course**: 20600 - Deep Learning for Computer Vision
**Institution**: Università Bocconi
**Academic Year**: Semester 1, 2025-2026
