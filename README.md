# Skin Cancer Classification — Transfer Learning + Explainable AI

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13%2B-orange?logo=tensorflow&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-EfficientNet--B0-red?logo=keras&logoColor=white)
![XAI](https://img.shields.io/badge/XAI-Grad--CAM%20%7C%20LIME-green)
![Dataset](https://img.shields.io/badge/Dataset-ISIC%20Skin%20Cancer-lightgrey)
![License](https://img.shields.io/badge/License-MIT-yellow)

**Binary skin lesion classification using Transfer Learning and Explainable AI.**  
Classifies dermoscopy images as *Malignant* or *Benign* with interpretable Grad-CAM heatmaps.

</div>

---

## Table of Contents

- [Overview](#overview)
- [Pipeline Architecture](#pipeline-architecture)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Training Strategy](#training-strategy)
- [Explainability — Grad-CAM & LIME](#explainability--grad-cam--lime)
- [Results](#results)
- [Project Structure](#project-structure)
- [Setup & Usage](#setup--usage)
- [Tech Stack](#tech-stack)

---

## Overview

This project tackles the binary classification of skin lesion images into **Malignant** vs **Benign** categories using:

- **Transfer Learning** — EfficientNet-B0 pretrained on ImageNet, fine-tuned on the ISIC dermoscopy dataset
- **Baseline Comparison** — Vanilla CNN trained from scratch to benchmark against the transfer learning approach
- **Explainable AI (XAI)** — Grad-CAM heatmaps and LIME explanations to identify which skin regions drive predictions
- **Two-phase fine-tuning** — Frozen base → full fine-tune with lower learning rate for stable convergence

Early detection of malignant skin lesions can be life-saving. This system aims to assist dermatologists by providing not just a prediction, but a visual explanation grounded in the image pixels.

---

## Pipeline Architecture

```mermaid
flowchart TD
    A[🗂️ ISIC Dataset\nKaggle] --> B[Data Loading\nPathlib glob · DataFrame]
    B --> C[Exploratory Data Analysis\nClass distribution · Sample images]
    C --> D[Preprocessing\n80/20 Train-Val Split · Stratified]
    D --> E[tf.data Pipeline\nBatch=32 · Prefetch · Cache]
    E --> F{Augmentation?}
    F -- Train only --> G[RandomFlip · RandomZoom]
    F -- Val / Test --> H[No augmentation]
    G --> I
    H --> I

    I[Model Training]
    I --> J[Baseline:\nVanilla CNN\nfrom scratch]
    I --> K[Transfer Learning:\nEfficientNet-B0\nImageNet weights]

    K --> L[Phase 1: Train Head\n10 epochs · LR=1e-4\nBase frozen]
    L --> M[Phase 2: Fine-tune All\n50 epochs · LR=1e-5\nEarly stopping patience=5]

    J --> N[Evaluation]
    M --> N

    N --> O[📊 Metrics\nConfusion Matrix · ROC · F1]
    N --> P[🔥 XAI\nGrad-CAM · LIME]

    O --> Q[training_artifacts/]
    P --> Q
```

---

## Dataset

| Split | Benign | Malignant | Total |
|-------|--------|-----------|-------|
| Train | 1,440  | 1,197     | 2,637 |
| Test  | 360    | 300       | 660   |

**Source:** [ISIC Skin Cancer Dataset (9 classes)](https://www.kaggle.com/datasets/nodoubttome/skin-cancer9-classesisic) on Kaggle.

The 9 original ISIC classes are mapped to a binary label:

```mermaid
flowchart LR
    subgraph Malignant
        M1[mel — Melanoma]
        M2[bcc — Basal Cell Carcinoma]
        M3[akiec — Actinic Keratosis]
    end
    subgraph Benign
        B1[nv — Nevus]
        B2[bkl — Benign Keratosis]
        B3[df — Dermatofibroma]
        B4[vasc — Vascular Lesion]
    end
    Malignant -->|label = 1| Binary[(Binary Labels)]
    Benign    -->|label = 0| Binary
```

Images are organized locally as:
```
data/
├── train/
│   ├── benign/       (1,440 images)
│   └── malignant/    (1,197 images)
└── test/
    ├── benign/       (360 images)
    └── malignant/    (300 images)
```

---

## Model Architecture

### Baseline — Vanilla CNN

A lightweight custom CNN trained from scratch as a performance baseline.

```mermaid
flowchart LR
    IN[Input\n224×224×3] --> C1[Conv2D 16\n3×3 ReLU]
    C1 --> P1[MaxPool2D\n2×2]
    P1 --> C2[Conv2D 8\n3×3 ReLU]
    C2 --> P2[MaxPool2D\n2×2]
    P2 --> FL[Flatten]
    FL --> DR[Dropout 0.2]
    DR --> D1[Dense 128\nReLU]
    D1 --> OUT[Dense 2\nSoftmax]
```

| Parameter | Value |
|-----------|-------|
| Total params | ~2.99M |
| Optimizer | Adam (lr=1e-3) |
| Loss | Categorical Crossentropy |
| Epochs | 50 (early stopping) |

---

### Transfer Learning — EfficientNet-B0

EfficientNet-B0 uses **compound scaling** to jointly scale depth, width, and resolution, offering superior accuracy-per-parameter compared to VGG or ResNet baselines.

```mermaid
flowchart LR
    IN[Input\n224×224×3] --> BASE[EfficientNet-B0\nImageNet weights\nfrozen in Phase 1]
    BASE --> GAP[GlobalAveragePooling2D]
    GAP --> DR[Dropout 0.3]
    DR --> OUT[Dense 2\nSoftmax\nMalignant · Benign]
```

| Parameter | Value |
|-----------|-------|
| Base model | EfficientNet-B0 (Keras Applications) |
| Input size | 224 × 224 × 3 |
| Head | GAP → Dropout(0.3) → Dense(2, softmax) |
| Phase 1 LR | 1e-4 (head only, 10 epochs) |
| Phase 2 LR | 1e-5 (full fine-tune, up to 50 epochs) |
| Early stopping | patience = 5 on `val_loss` |

---

## Training Strategy

```mermaid
sequenceDiagram
    participant D as Dataset
    participant M as Model
    participant CB as Callbacks

    D->>M: Phase 1 — Train head only (base frozen, 10 epochs, LR=1e-4)
    M->>CB: EarlyStopping · ModelCheckpoint
    CB-->>M: Save best weights

    D->>M: Phase 2 — Unfreeze all layers (fine-tune, LR=1e-5)
    M->>CB: EarlyStopping · ModelCheckpoint
    CB-->>M: Save best weights → training_artifacts/
```

**Data augmentation** is applied only during training to improve generalization:
- `RandomFlip` (horizontal + vertical)
- `RandomZoom` (±10% height and width)

---

## Explainability — Grad-CAM & LIME

### Grad-CAM

Gradient-weighted Class Activation Mapping (**Grad-CAM**) uses the gradients flowing into the last convolutional layer to produce a spatial heatmap highlighting the discriminative regions the model attended to.

```mermaid
flowchart TD
    IMG[Input Image\n224×224×3] --> MODEL[EfficientNet-B0]
    MODEL --> CONV[Last Conv Layer\nfeature maps]
    MODEL --> PRED[Softmax Predictions]
    PRED --> LOSS[Class Score\nfor predicted class]
    LOSS --> GRAD[∂Loss / ∂Feature Maps\nvia GradientTape]
    GRAD --> POOL[Global Average Pool\ngradients → weights]
    POOL --> WEIGHT[Weighted sum of\nfeature maps]
    WEIGHT --> RELU[ReLU → normalize]
    RELU --> HEAT[Heatmap 2D]
    HEAT --> OVL[Jet colormap overlay\non original image]
```

Output per image: **Original | Grad-CAM Heatmap | Overlay** — saved to `training_artifacts/`.

### LIME

**LIME** (Local Interpretable Model-agnostic Explanations) perturbs superpixel segments of the image and fits a local linear model to explain individual predictions — complementing Grad-CAM with segment-level attribution.

---

## Results

### Model Comparison

| Metric | Vanilla CNN | EfficientNet-B0 (GPU) |
|--------|:-----------:|:---------------------:|
| Test Accuracy | 81.4% | **88.6%** |
| Precision | 74.4% | **84.8%** |
| Recall | 90.0% | **91.3%** |
| F1 Score | 81.4% | **87.9%** |
| MCC | 0.642 | **0.774** |
| AUC (ROC) | 0.908 | **0.960** |
| Parameters | ~2.99M | ~4.05M |
| Interpretability | Grad-CAM | Grad-CAM + LIME |

> Detailed metrics and predictions are saved in `training_artifacts/`.

### Output Artifacts

| Artifact | Location |
|----------|----------|
| EfficientNet-B0 model (PyTorch GPU) | `training_artifacts/efficientnet_b0_gpu_best.pt` |
| Vanilla CNN model (Keras) | `training_artifacts/vanilla_cnn_model.keras` |
| EfficientNet-B0 test metrics | `training_artifacts/efficientnet_b0_gpu_test_metrics.csv` |
| EfficientNet-B0 test predictions | `training_artifacts/efficientnet_b0_gpu_test_predictions.csv` |
| EfficientNet-B0 training history | `training_artifacts/efficientnet_b0_gpu_history.csv` |
| Vanilla CNN test metrics | `training_artifacts/vanilla_cnn_test_metrics.csv` |
| Vanilla CNN training history | `training_artifacts/vanilla_cnn_training_history.csv` |
| Model comparison metrics | `training_artifacts/model_comparison_metrics.csv` |
| Label mapping | `training_artifacts/label_mapping.json` |

---

## Project Structure

```
skin-cancer-efficientnet-xai/
│
├── 📓 skin_cancer_TL_CNN.ipynb          ← Main project notebook (self-contained)
│
├── 📁 data/                              Dataset (populate before running)
│   ├── train/
│   │   ├── benign/      (1,440 images)
│   │   └── malignant/   (1,197 images)
│   └── test/
│       ├── benign/      (360 images)
│       └── malignant/   (300 images)
│
├── 📁 training_artifacts/               All training outputs
│   ├── efficientnet_b0_gpu_best.pt      EfficientNet-B0 trained model (PyTorch)
│   ├── vanilla_cnn_model.keras          Vanilla CNN trained model (Keras)
│   ├── efficientnet_b0_gpu_history.csv  EfficientNet training history
│   ├── vanilla_cnn_training_history.csv Vanilla CNN training history
│   ├── efficientnet_b0_gpu_test_metrics.csv
│   ├── efficientnet_b0_gpu_test_predictions.csv
│   ├── vanilla_cnn_test_metrics.csv
│   ├── model_comparison_metrics.csv     Side-by-side model comparison
│   └── label_mapping.json
│
├── 📄 requirements.txt                  Python dependencies
└── 📄 README.md
```

---

## Setup & Usage

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

**`requirements.txt`** includes:
```
tensorflow>=2.13,<2.19
numpy
pandas
matplotlib
seaborn
scikit-learn
pillow
kagglehub
```

### 2. Prepare the Dataset

Ensure the following folder structure exists (images already present locally):

```
data/train/benign/
data/train/malignant/
data/test/benign/
data/test/malignant/
```

Or download from Kaggle:
```bash
pip install kagglehub
python -c "import kagglehub; kagglehub.dataset_download('nodoubttome/skin-cancer9-classesisic')"
```

### 3. Run the Notebook

Open and run **`skin_cancer_TL_CNN.ipynb`** cell by cell in Jupyter or VS Code.

The notebook is fully self-contained and covers:

```mermaid
flowchart LR
    A[✅ Dependency check] --> B[📂 Load data]
    B --> C[📊 EDA]
    C --> D[⚙️ Preprocessing]
    D --> E[🧠 Vanilla CNN\nbaseline]
    E --> F[🚀 EfficientNet-B0\ntransfer learning]
    F --> G[📈 Evaluation\nmetrics]
    G --> H[🔥 Grad-CAM\nXAI]
    H --> I[🟩 LIME\nXAI]
```

### 4. Outputs

All outputs are automatically saved to `training_artifacts/`:
- **EfficientNet-B0 model** → `training_artifacts/efficientnet_b0_gpu_best.pt`
- **Vanilla CNN model** → `training_artifacts/vanilla_cnn_model.keras`
- **Metrics & predictions** → `training_artifacts/*.csv`
- **Model comparison** → `training_artifacts/model_comparison_metrics.csv`

---

## Tech Stack

| Component | Tool / Library |
|-----------|---------------|
| Language | Python 3.10+ |
| Deep Learning Framework | TensorFlow 2.13+ / Keras |
| Pretrained Model | EfficientNet-B0 (ImageNet) |
| Data Pipeline | `tf.data.Dataset` |
| XAI — Grad-CAM | Manual `tf.GradientTape` implementation |
| XAI — LIME | `lime` + `scikit-image` |
| Evaluation | `scikit-learn` (confusion matrix, ROC, F1, MCC) |
| Visualization | Matplotlib · Seaborn |
| Dataset | ISIC via `kagglehub` |
| Notebook | Jupyter / VS Code |

---

## How Grad-CAM Works — Visually

```
Input Image          Grad-CAM Heatmap        Overlay
┌─────────────┐      ┌─────────────┐      ┌─────────────┐
│             │      │  ░░░▓▓▓██   │      │  ░░░▓▓▓██   │
│  skin       │  ──► │  ░▓▓████▓   │  ──► │  blended    │
│  lesion     │      │  ░░▓▓███░   │      │  image      │
│             │      │  ░░░░░░░░   │      │             │
└─────────────┘      └─────────────┘      └─────────────┘
                     red = high attention  jet colormap
```

Darker red regions indicate the areas the model weighted most heavily when making its malignant/benign decision — enabling clinicians to verify the model is attending to the lesion itself, not background artifacts.

---

<div align="center">
<sub>Built with TensorFlow · EfficientNet-B0 · Grad-CAM · LIME · ISIC Dataset</sub>
</div>
