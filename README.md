# 🦴 FractureMamba-ViT: Dual-Stream Hybrid Architecture for Bone Fracture Classification

> **Hackathon**: Bone Fracture Classification Hackathon 2025  
> **Organized by**: Kamand Bioengineering Group, IIT Mandi  
> **Architecture**: Vision Mamba (SSM) + Swin Transformer + Cross-Attention Fusion

---

## 🏆 Architecture Overview

**FractureMamba-ViT** is a novel dual-stream hybrid architecture that combines:

1. **Vision Mamba (Stream 1)** — State space model with bidirectional scanning for efficient long-range sequence modeling of fracture patterns
2. **Swin Transformer (Stream 2)** — Shifted window self-attention for hierarchical spatial feature extraction
3. **Cross-Attention Fusion** — Bidirectional cross-attention with learned gating to combine both streams
4. **Diffusion Augmentation** — Conditional DDPM generates synthetic X-rays for class balancing

```
Input X-ray (224×224)
        │
   ┌────┴────┐
   │         │
Vision    Swin
Mamba   Transformer
(SSM)   (Attention)
   │         │
   └────┬────┘
        │
  Cross-Attention
  Fusion + Gating
        │
   MLP Classifier
        │
  Fracture / Not Fractured
```

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Train the Model
```bash
# Full training (100 epochs)
python train.py --config config.yaml

# Quick test (2 epochs)
python train.py --config config.yaml --debug

# 5-fold cross-validation
python train.py --config config.yaml --cv
```

### 3. Evaluate
```bash
python evaluate.py --config config.yaml --tta
```

### 4. Generate Explainability Visualizations
```bash
python explainability.py --config config.yaml --num-samples 5
```

## 📁 Project Structure

```
├── config.yaml              # All hyperparameters
├── data_loader.py           # Dataset, transforms, MixUp/CutMix, TTA
├── mamba_module.py          # Vision Mamba (pure PyTorch, no CUDA kernels)
├── model.py                 # FractureMamba-ViT + FocalLoss
├── diffusion_augment.py     # Conditional DDPM for data augmentation
├── train.py                 # Training loop (AMP, SWA, gradient accum)
├── evaluate.py              # Metrics, final_results.csv generation
├── explainability.py        # Grad-CAM, attention maps, state viz
├── utils.py                 # Utilities, checkpointing, plotting
├── requirements.txt         # Pinned dependencies
├── README.md                # This file
└── TEAM.txt                 # Team information
```

## ⚙️ Key Training Features

| Feature | Details |
|---|---|
| Mixed Precision (FP16) | 50% memory reduction via `torch.cuda.amp` |
| Gradient Accumulation | Batch 8 × 4 steps = effective batch 32 |
| Focal Loss | γ=2.0, auto-weighted α for class imbalance |
| Label Smoothing | ε=0.1 for regularization |
| MixUp + CutMix | Batch-level augmentation (α=0.4, α=1.0) |
| CLAHE | X-ray contrast enhancement |
| SWA | Stochastic Weight Averaging (last 20 epochs) |
| Cosine Annealing | With warm restarts (T₀=10, T_mult=2) |
| Early Stopping | Patience=20 epochs |
| 5-Fold Stratified CV | Robust generalization estimate |
| Test-Time Augmentation | 5 augmented views averaged |

## 📊 Metrics Generated

- `final_results.csv` — Overall accuracy, per-class P/R/F1, AUC-ROC, inference time
- `model_performance_analysis.csv` — Epoch-by-epoch loss/accuracy/gap
- `confusion_matrix.png` — Heatmap visualization
- `training_curves.png` — Loss, accuracy, overfitting gap, LR schedule

## 🔍 Explainability Outputs

- **Grad-CAM saliency maps** — Highlights fracture regions driving classification
- **Mamba attention maps** — Token importance from the SSM stream
- **Mamba state visualization** — Hidden state evolution, sequential coherence
- **Prediction comparison grid** — Quick visual of model predictions vs ground truth

## 📝 Notes

- **Pre-trained weights**: Uses ImageNet-pretrained Swin Transformer (allowed per rules). No fracture-specific pre-training.
- **Mamba implementation**: Pure PyTorch selective scan — works on Windows/Linux without CUDA kernels.
- **Hardware**: Optimized for 6GB VRAM (RTX 3050) with FP16 + gradient accumulation.

## 📚 References

- [Vision Mamba (Vim)](https://arxiv.org/abs/2401.09417)
- [Swin Transformer](https://arxiv.org/abs/2103.14030)
- [MedMamba](https://arxiv.org/abs/2403.03849)
- [Focal Loss](https://arxiv.org/abs/1708.02002)
