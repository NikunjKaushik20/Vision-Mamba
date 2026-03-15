<div align="center">

# 🦴 FractureMamba-ViT

### Dual-Stream Hybrid Deep Learning Architecture for Bone Fracture Detection & Localization

[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![PyTorch 2.1+](https://img.shields.io/badge/PyTorch-2.1+-ee4c2c.svg)](https://pytorch.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Accuracy](https://img.shields.io/badge/Test%20Accuracy-99.8%25-brightgreen.svg)](#-model-performance)
[![AUC-ROC](https://img.shields.io/badge/AUC--ROC-1.000-brightgreen.svg)](#-model-performance)

**An AI-powered bone fracture detection system combining Vision Mamba state-space models with Swin Transformer attention, enhanced by YOLOv8 localization and an interactive web interface with an AI radiologist assistant.**

[Architecture](#-architecture) · [Quick Start](#-quick-start) · [Web UI](#-web-interface) · [Results](#-model-performance) · [Project Structure](#-project-structure)

</div>

---

## 📖 Table of Contents

- [Overview](#-overview)
- [Architecture](#-architecture)
- [Model Performance](#-model-performance)
- [Quick Start](#-quick-start)
- [Training](#-training)
- [Evaluation](#-evaluation)
- [Web Interface](#-web-interface)
- [Explainability & Visualizations](#-explainability--visualizations)
- [Project Structure](#-project-structure)
- [Configuration](#-configuration)
- [Tech Stack](#-tech-stack)
- [Team](#-team)
- [References](#-references)

---

## 🔬 Overview

**FractureMamba-ViT** is a multi-layer AI system for automated bone fracture detection from X-ray images. It is designed to assist radiologists by providing:

- **Binary classification** — Fractured vs. Not Fractured (99.8% accuracy)
- **Fracture localization** — Bounding box detection using YOLOv8
- **Explainability** — Grad-CAM saliency maps and Mamba state-space visualizations
- **AI Chat Assistant** — GPT-powered interactive assistant for clinical Q&A
- **Real-world robustness** — Augmentation pipeline simulating WhatsApp compression, camera blur, and screen photography

The system uses a **three-layer ensemble** approach:

| Layer | Model | Role |
|-------|-------|------|
| **Layer 1** | FractureMamba-ViT (Vision Mamba ∨ Swin Transformer) | Binary fracture classification via OR-gate consensus |
| **Layer 2** | YOLOv8 (fine-tuned on FracAtlas) | Fracture localization & bounding box, OR-gated with Layer 1 |
| **Layer 3** | Swin-Tiny Classifier + GPT-4o-mini | Fracture type classification (e.g., transverse, oblique, spiral) |

---

## 🧠 Architecture

FractureMamba-ViT uses an **OR-gate ensemble** strategy to maximize recall — in medical imaging, missing a fracture (false negative) is far more dangerous than a false alarm. If **any** model detects a fracture, the system flags it.

```
                              Input X-ray (224×224)
                                      │
              ┌───────────────────────┼───────────────────────┐
              │                       │                       │
     ┌────────┴────────┐     ┌────────┴────────┐     ┌────────┴────────┐
     │  Vision Mamba   │     │ Swin Transformer│     │     YOLOv8      │
     │     (SSM)       │     │   (Attention)   │     │   (FracAtlas)   │
     │                 │     │                 │     │                 │
     │  Bi-dir SSM     │     │ Shifted Window  │     │  Object Det.    │
     │  4 blocks       │     │ Self-Attention  │     │  + Bounding     │
     │  d=192          │     │ Pretrained      │     │    Box          │
     └────────┬────────┘     └────────┬────────┘     └────────┬────────┘
              │                       │                       │
        Fractured?              Fractured?              Box detected?
         (binary)                (binary)               (detection)
              │                       │                       │
              └───────────┬───────────┘                       │
                          │                                   │
                  ┌───────┴───────┐                           │
                  │   OR Gate     │  ← Layer 1                │
                  │ (Mamba ∨ Swin │                           │
                  │  = fracture)  │                           │
                  └───────┬───────┘                           │
                          │                                   │
                          └───────────────┬───────────────────┘
                                          │
                                  ┌───────┴───────┐
                                  │   OR Gate     │  ← Layer 2
                                  │ (ViT ∨ YOLO  │
                                  │  = fracture)  │
                                  └───────┬───────┘
                                          │
                            ┌─────────────┴─────────────┐
                            │                           │
                      ✅ Fractured                ❌ Not Fractured
                            │                        (stop)
                            │ 
            ┌───────────────┼
            │               │               
            ▼               ▼              
   ┌────────┴────────┐ ┌────┴────────┐ 
   │   Layer 3:      │ │  Grad-CAM   │ 
   │  Fracture Type  │ │  Heatmap    │
   │  Classification │ │             │
   │                 │ │ Saliency    │
   │  Swin-Tiny +    │ │ overlay on  │
   │  GPT-4o-mini    │ │ X-ray image │
   └────────┬────────┘ └──────┬──────┘ 
            │                 │               
     Fracture Type     Explainability
      + Confidence      (Jet cmap)      
```

### Key Components

| Component | Description |
|-----------|-------------|
| **Vision Mamba (Stream 1)** | Bidirectional state-space model with S6 selective scan for efficient long-range sequence modeling of fracture patterns. Pure PyTorch implementation — no CUDA kernels required. |
| **Swin Transformer (Stream 2)** | ImageNet-pretrained shifted window self-attention (`swin_tiny_patch4_window7_224`) for hierarchical local spatial feature extraction. |
| **YOLOv8 (Layer 2)** | Fine-tuned YOLOv8-nano on FracAtlas dataset for fracture localization. Provides bounding box coordinates and detection confidence. |
| **OR-Gate Ensemble** | If **any** model (Mamba, Swin, or YOLO) detects a fracture, the system flags it as fractured — maximizing recall for patient safety. |
| **Fracture Type Classifier (Layer 3)** | Swin-Tiny classifier ensembled with GPT-4o-mini for fracture type classification (transverse, oblique, spiral, comminuted, etc.). Only runs when fracture is confirmed. |
| **Focal Loss** | γ=2.0 with auto-weighted α and label smoothing (ε=0.1) for class imbalance handling. |

### Why OR-Gate Instead of Projection Fusion?

- **Patient safety first** — Projection-based fusion *averages* the two streams' outputs. If one model detects a fracture but the other doesn't, the fused signal can be diluted, potentially **missing the fracture**. With an OR gate, if either model flags it, it's flagged.
- **Interpretability** — With fusion, you get one opaque blended prediction. With OR-gate ensemble, you know *exactly* which model detected the fracture (ViT, YOLO, or both), enabling transparent clinical reporting.
- **Zero overhead** — The OR gate is a simple boolean check. Inference time is dominated by the model forward passes, so switching from cross-attention fusion to OR gate had **no impact on latency** while significantly improving recall and explainability.
- **Modularity** — Each model can be independently trained, updated, or swapped without retraining a shared fusion layer.

---

## 📊 Model Performance

### Test Set Results (Bone Fracture Binary Classification)

| Metric | Score |
|--------|-------|
| **Overall Accuracy** | **99.8%** |
| **Precision (Macro)** | 99.8% |
| **Recall (Macro)** | 99.8% |
| **F1-Score (Macro)** | 99.8% |
| **AUC-ROC** | **1.000** |
| **Inference Time** | 96.3 ms/image |
| **Model Size** | 127.3 MB |

### Per-Class Performance

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| Fractured | 0.996 | 1.000 | 0.998 | 238 |
| Not Fractured | 1.000 | 0.996 | 0.998 | 268 |

### Confusion Matrix

|  | Predicted: Fractured | Predicted: Not Fractured |
|--|---------------------|--------------------------|
| **Actual: Fractured** | 238 | 0 |
| **Actual: Not Fractured** | 1 | 267 |

---

## 🚀 Quick Start

### Prerequisites

- **Python** 3.10+
- **CUDA** 11.8+ (recommended, CPU works but is slower)
- **Node.js** 18+ (for the web interface only)
- **GPU** with 6GB+ VRAM (optimized for RTX 3050)

### 1. Clone the Repository

```bash
git clone https://github.com/NikunjKaushik20/Vision-Mamba.git
cd Vision-Mamba
```

### 2. Install Python Dependencies

```bash
# Create and activate virtual environment (recommended)
python -m venv venv

# Windows
venv\Scripts\activate

# Linux / macOS
source venv/bin/activate

# Install core dependencies
pip install -r requirements.txt

# Install YOLO support (for fracture localization)
pip install ultralytics

# Install web UI backend dependencies
pip install -r web_ui/backend/requirements.txt
```

### 3. Prepare the Dataset

Download the **Bone Fracture Binary Classification** dataset and place it in the `data/` directory:

```
data/
└── Bone_Fracture_Binary_Classification/
    └── Bone_Fracture_Binary_Classification/
        ├── train/
        │   ├── fractured/
        │   └── not fractured/
        ├── val/
        │   ├── fractured/
        │   └── not fractured/
        └── test/
            ├── fractured/
            └── not fractured/
```

Update the `data.root_dir` in `config.yaml` to point to your dataset path.

### 4. Train the Model

```bash
cd src

# Full training (100 epochs)
python train.py --config ../config.yaml

# Quick test run (2 epochs, small batch)
python train.py --config ../config.yaml --debug

# 5-fold stratified cross-validation
python train.py --config ../config.yaml --cv
```

### 5. Evaluate

```bash
# Evaluate with Test-Time Augmentation (TTA)
python evaluate.py --config ../config.yaml --tta
```

### 6. Single Image Inference

```bash
python test_single.py --img /path/to/xray.jpg
```

---

## 🏋️ Training

### Training Pipeline Features

| Feature | Details |
|---------|---------|
| **Mixed Precision (FP16)** | ~50% memory reduction via `torch.cuda.amp` |
| **Gradient Accumulation** | Batch 8 × 4 steps = effective batch size 32 |
| **Focal Loss** | γ=2.0, auto-weighted α for class imbalance |
| **Label Smoothing** | ε=0.1 for regularization |
| **MixUp + CutMix** | Batch-level augmentation (α=0.4, α=1.0, p=0.5) |
| **CLAHE** | X-ray contrast enhancement (clip=2.0, grid=8×8) |
| **SWA** | Stochastic Weight Averaging (last 20 epochs, lr=5e-5) |
| **Cosine Annealing** | With warm restarts (T₀=10, T_mult=2) |
| **Warmup** | 5 epochs, linear warmup from 1e-6 to 1e-4 |
| **Early Stopping** | Patience=20 epochs, monitoring val accuracy |
| **Gradient Clipping** | Max norm=1.0 |
| **Weighted Sampling** | Inverse class frequency for balanced batches |

### Data Augmentation Pipeline

**Standard Medical Imaging Augmentations:**
- Horizontal flip, rotation (±15°), affine transforms
- Color jitter, Gaussian blur, noise, elastic transforms
- CoarseDropout (random erasing)

**Real-World Robustness Augmentations** (simulating phone photos of X-rays):
- JPEG compression (quality 15-50, p=0.6) — WhatsApp/messaging compression
- Perspective transform (p=0.5) — Phone held at angle
- Motion blur (p=0.3) — Shaky hands
- ISO noise (p=0.4) — Cheap smartphone sensors
- Sun flare (p=0.2) — Room lights glaring on screens
- Grid dropout (p=0.3) — Moiré patterns from monitor LCD

### YOLO Fracture Localization Training

```bash
# Fine-tune YOLOv8n on FracAtlas dataset
python scripts/train_yolo.py
```

Trains YOLOv8-nano for 100 epochs on the FracAtlas fracture localization dataset with mosaic augmentation, cosine LR scheduling, and validation.

---

## 📈 Evaluation

The evaluation pipeline computes comprehensive metrics and generates publication-ready outputs:

```bash
# Standard evaluation
python src/evaluate.py --config config.yaml

# With Test-Time Augmentation (5 augmented views averaged)
python src/evaluate.py --config config.yaml --tta

# Specify a custom checkpoint
python src/evaluate.py --config config.yaml --checkpoint checkpoints/checkpoint_robust_best.pth
```

### Generated Outputs

| File | Description |
|------|-------------|
| `results/final_results.csv` | Accuracy, Precision, Recall, F1, AUC-ROC, inference time, model size |
| `results/model_performance_analysis.csv` | Epoch-by-epoch train/val loss, accuracy, overfitting gap, LR |
| `results/confusion_matrix.png` | Heatmap visualization of the confusion matrix |
| `results/training_curves.png` | Loss curves, accuracy curves, overfitting gap, LR schedule |

### Test-Time Augmentation (TTA)

TTA averages predictions across 5 augmented views of each test image:
1. Original
2. Horizontal flip
3. +5° rotation
4. -5° rotation
5. 1.1× center crop

---

## 🌐 Web Interface

The project includes a full-stack web application for interactive fracture analysi

### Web UI Features

- **Drag & drop X-ray upload** (DICOM, PNG, JPG supported)
- **Live camera capture** for photographing physical X-rays
- **4-panel analysis view:**
  1. Original X-ray
  2. YOLO fracture localization with bounding box
  3. Grad-CAM saliency heatmap
  4. Mamba state-space visualization
- **AI Chat Assistant** — Ask questions about the scan results
- **Confidence scoring** — Visual confidence bar with ensemble verdict
- **Dark mode** — Premium dark interface with glassmorphism effects
- **Responsive design** — Works on desktop and mobile

### Starting the Web UI

#### Step 1: Start the Backend

```bash
cd web_ui/backend

# Create a .env file with your OpenAI API key (optional, for chat only)
echo OPENAI_API_KEY=sk-your-key-here > .env

# Install backend dependencies
pip install -r requirements.txt

# Start the FastAPI server
python main.py
# Server starts at http://localhost:8000
```

#### Step 2: Start the Frontend

```bash
cd web_ui/frontend

# Install dependencies
npm install

# Start the development server
npm run dev
# App opens at http://localhost:5173
```

### API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/predict` | Upload an X-ray image for analysis. Returns classification, confidence, GradCAM, Mamba viz, YOLO localization, and ensemble verdict. |
| `POST` | `/chat` | Send a message to the AI assistant. Context from the last scan is automatically injected. |
| `GET` | `/health` | Health check. Returns model loading status and YOLO availability. |

### OR-Gate Ensemble Decision Logic

| ViT Prediction | YOLO Detection | Ensemble Verdict | Localization |
|----------------|---------------|-------------------|--------------|
| Fractured | Box found | ✅ Confirmed Fracture | YOLO bounding box |
| Fractured | No box | ✅ Confirmed Fracture | GradCAM fallback |
| Not Fractured | Box found | ⚠️ Override → Fractured | YOLO bounding box |
| Not Fractured | No box | ✅ Not Fractured | N/A |

---

## 🔍 Explainability & Visualizations

Generate all explainability outputs:

```bash
python src/explainability.py --config config.yaml --num-samples 5
```

### Visualization Types

| Visualization | Description |
|---------------|-------------|
| **Grad-CAM Saliency Maps** | Highlights image regions that most influenced the classification decision, using gradients from the Swin Transformer stream. |
| **Mamba Attention Maps** | Token importance (L2 norm) from the Vision Mamba SSM stream, reshaped into a spatial heatmap to show which patch positions the model attends to. |
| **Mamba State Evolution** | Plots of hidden state norm and cosine similarity across the patch sequence — reveals sequential patterns the SSM detects (e.g., discontinuities at fracture lines). |
| **Prediction Comparison Grid** | Side-by-side grid of model predictions vs. ground truth for rapid visual quality assessment. |

All outputs are saved to `explainability_outputs/`.

---

## 📁 Project Structure

```
Vision-Mamba/
│
├── config.yaml                          # Main configuration (hyperparameters, paths, augmentation)
├── requirements.txt                     # Python dependencies
├── README.md                            # This file
├── TEAM.txt                             # Team information
│
├── src/                                 # Core ML source code
│   ├── model.py                         # FractureMamba-ViT architecture + FocalLoss
│   ├── mamba_module.py                  # Vision Mamba (S6 SSM, bidirectional blocks)
│   ├── train.py                         # Training loop (AMP, SWA, CV, gradient accum)
│   ├── evaluate.py                      # Evaluation metrics, final_results.csv, confusion matrix
│   ├── data_loader.py                   # Dataset, transforms, MixUp/CutMix, TTA
│   ├── explainability.py                # Grad-CAM, attention maps, Mamba state visualization
│   ├── utils.py                         # Utilities (config, seeding, checkpointing, plotting)
│   ├── test_single.py                   # Single image inference script
│   ├── evaluate_real_life.py            # Real-life photo evaluation (WhatsApp images)
│   ├── auto_crop.py                     # Automatic X-ray region cropping utility
│   ├── diffusion_augment.py             # Conditional DDPM for synthetic X-ray generation
│   ├── fine_tune_real_world.py          # Fine-tuning on real-world noisy images
│   └── colab_train.py                   # Google Colab training script (standalone)
│
├── scripts/                             # Utility scripts
│   └── train_yolo.py                    # YOLOv8 fine-tuning for fracture localization
│
├── configs/                             # Additional config files
│   ├── fracatlas_yolo.yaml              # YOLOv8 dataset config (FracAtlas)
│   └── fracture_type_config.yaml        # Fracture type classification config
│
├── web_ui/                              # Full-stack web application
│   ├── backend/                         # FastAPI backend
│   │   ├── main.py                      # API server (predict, chat, health endpoints)
│   │   ├── inference_wrapper.py         # Model manager (ViT + YOLO + GradCAM + Mamba viz)
│   │   ├── chat_assistant.py            # OpenAI-powered AI radiologist assistant
│   │   ├── requirements.txt             # Backend Python dependencies
│   │   └── .env                         # Environment variables (OPENAI_API_KEY)
│   │
│   └── frontend/                        # React + Vite + TailwindCSS frontend
│       ├── src/
│       │   ├── components/
│       │   │   ├── LandingPage.jsx       # Landing/hero page
│       │   │   ├── ScannerApp.jsx        # Main scanner interface
│       │   │   ├── Dashboard.jsx         # Dashboard layout
│       │   │   ├── AnalysisWorkspace.jsx  # 4-panel analysis view
│       │   │   ├── ChatAssistant.jsx      # AI chat sidebar
│       │   │   ├── Sidebar.jsx            # Navigation sidebar
│       │   │   ├── UploadZone.jsx         # File upload component
│       │   │   └── ResultsDashboard.jsx   # Results summary
│       │   ├── App.jsx                   # App router
│       │   ├── index.css                 # Global styles & design tokens
│       │   └── main.jsx                  # React entry point
│       └── package.json                  # Node.js dependencies
│
├── checkpoints/                         # Trained model weights (Git LFS)
│   ├── checkpoint_robust_best.pth       # Best robust model (primary)
│   ├── checkpoint_best.pth              # Best validation accuracy
│   ├── checkpoint.pth                   # Latest checkpoint
│   └── checkpoint_swa.pth               # SWA-averaged weights
│
├── results/                             # Evaluation results
│   ├── final_results.csv                # Comprehensive metrics table
│   ├── model_performance_analysis.csv   # Epoch-by-epoch analysis
│   ├── confusion_matrix.png             # Confusion matrix heatmap
│   └── training_curves.png              # Training/validation curves
│
├── explainability_outputs/              # Generated visualizations
│   ├── attention_map_*.png              # Mamba attention heatmaps
│   ├── mamba_state_*.png                # SSM state evolution plots
│   └── prediction_grid.png             # Prediction comparison grid
│
├── docs/                                # Documentation
│   ├── images/                          # Figures for docs and README
│   └── reports/                         # Technical reports (LaTeX + PDF)
│
├── weights/                             # Pre-trained base weights
├── data/                                # Datasets (gitignored)
└── runs/                                # YOLO training runs
```

---

## ⚙️ Configuration

All hyperparameters are centralized in `config.yaml`:

### Key Configuration Sections

| Section | Parameters |
|---------|-----------|
| **`data`** | `root_dir`, `image_size` (224), `num_classes` (2), `class_names` |
| **`model.mamba`** | `embed_dim` (192), `depth` (4), `patch_size` (16), `d_state` (8), `d_conv` (4) |
| **`model.swin`** | `model_name` (swin_tiny), `pretrained` (true), `drop_path_rate` (0.2) |
| **`model.fusion`** | `dim` (384), `num_heads` (8), `dropout` (0.1) |
| **`training`** | `epochs` (100), `batch_size` (8), `lr` (1e-4), `weight_decay` (0.05) |
| **`training.swa`** | `enabled` (true), `start_epoch` (80), `lr` (5e-5) |
| **`augmentation`** | CLAHE, flips, rotation, affine, color jitter, MixUp, CutMix |
| **`augmentation.real_world`** | JPEG compression, perspective, motion blur, ISO noise, sun flare |
| **`evaluation`** | TTA (`num_augments`: 5), cross-validation (`n_folds`: 5) |

---

## 🛠️ Tech Stack

### Machine Learning
| Technology | Usage |
|-----------|-------|
| **PyTorch 2.1+** | Core deep learning framework |
| **timm** | Swin Transformer pretrained models |
| **einops** | Tensor rearrangement operations |
| **Albumentations** | Image augmentation pipeline |
| **Ultralytics YOLOv8** | Fracture localization |
| **scikit-learn** | Metrics and cross-validation |
| **OpenCV** | Image I/O and preprocessing |
| **Matplotlib / Seaborn** | Visualization and plotting |

### Web Application
| Technology | Usage |
|-----------|-------|
| **FastAPI** | Backend REST API |
| **React 19** | Frontend UI framework |
| **Vite** | Frontend build tool |
| **TailwindCSS** | Styling |
| **OpenAI API** | AI Chat Assistant (GPT-3.5-turbo) |

---

## 📝 Important Notes

- **Pre-trained weights**: Uses ImageNet-pretrained Swin Transformer via `timm`. No fracture-specific pre-training of the backbone.
- **Mamba implementation**: Pure PyTorch selective scan with chunked parallelism — works on **Windows, Linux, and macOS** without custom CUDA kernels.
- **Hardware**: Optimized for 6GB VRAM GPUs (RTX 3050/3060) using FP16 mixed precision + gradient accumulation.
- **Checkpoints**: Model weights (`.pth` files) are tracked via Git LFS. Run `git lfs pull` after cloning.
- **Chat Assistant**: Requires an OpenAI API key set in `web_ui/backend/.env`. The chat feature is optional — the main detection pipeline works without it.

---

## 👥 Team

**Team Ctrl Alt Elite** — Maharaja Agrasen Institute of Technology, Delhi

| Member | Contact |
|--------|---------|
| Nikunj Kaushik | nikunjkaushik28@gmail.com |
| Pulkit Rustagi | pulkitrustagi8@gmail.com |
| Tavish Agarwal | tavishagarwal07@gmail.com |
| Varun Pathak | varunpathakvp15@gmail.com |

---

## 📚 References

1. Zhu, L., et al. **"Vision Mamba: Efficient Visual Representation Learning with Bidirectional State Space Model."** *arXiv:2401.09417* (2024). [[Paper]](https://arxiv.org/abs/2401.09417)
2. Liu, Z., et al. **"Swin Transformer: Hierarchical Vision Transformer using Shifted Windows."** *ICCV 2021*. [[Paper]](https://arxiv.org/abs/2103.14030)
3. Yue, Y., & Li, Z. **"MedMamba: Vision Mamba for Medical Image Classification."** *arXiv:2403.03849* (2024). [[Paper]](https://arxiv.org/abs/2403.03849)
4. Lin, T.-Y., et al. **"Focal Loss for Dense Object Detection."** *ICCV 2017*. [[Paper]](https://arxiv.org/abs/1708.02002)
5. Gu, A., & Dao, T. **"Mamba: Linear-Time Sequence Modeling with Selective State Spaces."** *arXiv:2312.00752* (2023). [[Paper]](https://arxiv.org/abs/2312.00752)

---

<div align="center">

**Built with ❤️ for advancing AI-assisted radiology**

*⚕️ Disclaimer: This tool is designed for research and educational purposes. It is not a certified medical device. All AI-generated analysis should be verified by a qualified radiologist.*

</div>
