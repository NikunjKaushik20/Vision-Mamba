"""
train_fracture_type.py
======================
Layer 3 of the FractureMamba-ViT pipeline:
Swin-Tiny fine-tuned on Bone Break Classification dataset (10 classes).

Usage:
    python train_fracture_type.py                          # trains with default config
    python train_fracture_type.py --config configs/fracture_type_config.yaml
    python train_fracture_type.py --eval-only              # evaluate saved best.pth
    python train_fracture_type.py --epochs 30              # override epochs
"""

import os
import sys
import json
import time
import argparse
import random
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler, Subset

import timm
from PIL import Image
from pathlib import Path
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import classification_report, confusion_matrix, f1_score, roc_auc_score
from torchvision import transforms
from tqdm import tqdm
import yaml
import warnings
warnings.filterwarnings("ignore")


# ─── Reproducibility ──────────────────────────────────────────────────────────

def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device():
    if torch.cuda.is_available():
        dev = torch.device("cuda")
        print(f"[DEVICE] GPU: {torch.cuda.get_device_name(0)}")
    else:
        dev = torch.device("cpu")
        print("[DEVICE] CPU (training will be slower)")
    return dev


# ─── Config ───────────────────────────────────────────────────────────────────

def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


DEFAULT_CONFIG = {
    "seed": 42,
    "deterministic": False,
    "data": {
        "root_dir": "data/Bone Break Classification/Bone Break Classification",
        "image_size": 224, "num_workers": 4,
        "train_split": 0.70, "val_split": 0.15,
    },
    "training": {
        "epochs": 60, "batch_size": 32, "gradient_accumulation_steps": 1,
        "mixed_precision": True, "gradient_clip_norm": 1.0, "label_smoothing": 0.1,
        "optimizer": {"lr": 3e-4, "weight_decay": 0.05, "betas": [0.9, 0.999]},
        "scheduler": {"T_0": 20, "T_mult": 1, "eta_min": 1e-6},
        "focal_loss": {"gamma": 2.0},
        "warmup": {"epochs": 5, "start_lr": 1e-6},
        "early_stopping": {"patience": 15, "min_delta": 0.1},
    },
    "model": {
        "name": "swin_tiny_patch4_window7_224", "pretrained": True,
        "drop_path_rate": 0.2,
        "head": {"hidden_dim": 256, "dropout": 0.35},
    },
    "augmentation": {
        "train": {
            "mixup": {"enabled": True, "alpha": 0.4},
            "mixup_cutmix_prob": 0.5,
        },
    },
    "paths": {
        "checkpoint_dir": "runs/fracture_type",
        "results_dir": "results/fracture_type",
    },
}


# ─── Dataset ──────────────────────────────────────────────────────────────────

class FractureTypeDataset(Dataset):
    """Loads images from a flat folder tree: root/ClassName/images.*"""

    EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}

    def __init__(self, root_dir, indices, all_paths, all_labels, transform=None):
        self.transform = transform
        self.paths = [all_paths[i] for i in indices]
        self.labels = [all_labels[i] for i in indices]

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, self.labels[idx]


def scan_dataset(root_dir):
    """
    Scan dataset with structure:
        root/ClassName/images.*           (flat)
        root/ClassName/Train/images.*     (with Train/Test subfolders)
        root/ClassName/Test/images.*
    All images per class are pooled — we do our own stratified split.
    """
    root = Path(root_dir)
    SPLIT_DIRS = {"train", "test", "val", "valid", "validation"}

    # Collect class folders (exclude hidden dirs, file-only root)
    class_dirs = sorted([d for d in root.iterdir() if d.is_dir() and not d.name.startswith(".")])
    classes = [d.name for d in class_dirs]
    cls2idx = {c: i for i, c in enumerate(classes)}

    paths, labels = [], []
    for cls_dir in class_dirs:
        cls_idx = cls2idx[cls_dir.name]
        found = 0

        # Check for Train/Test subfolders
        subdirs = [d for d in cls_dir.iterdir() if d.is_dir()]
        split_subdirs = [d for d in subdirs if d.name.lower() in SPLIT_DIRS]

        if split_subdirs:
            # ClassName/Train/*.jpg  ClassName/Test/*.jpg
            search_dirs = split_subdirs
        else:
            # Flat: ClassName/*.jpg
            search_dirs = [cls_dir]

        for sdir in search_dirs:
            for ext in FractureTypeDataset.EXTENSIONS:
                for p in sdir.glob(f"**/*{ext}"):
                    paths.append(str(p))
                    labels.append(cls_idx)
                    found += 1

        print(f"  {cls_dir.name}: {found} images")

    return paths, labels, classes, cls2idx


def get_transforms(config, split="train"):
    size = config["data"]["image_size"]
    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]

    if split == "train":
        aug_cfg = config.get("augmentation", {}).get("train", {})
        jitter  = aug_cfg.get("color_jitter", {})
        return transforms.Compose([
            transforms.RandomResizedCrop(size, scale=(0.7, 1.0), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(
                brightness=jitter.get("brightness", 0.3),
                contrast=jitter.get("contrast", 0.3),
                saturation=jitter.get("saturation", 0.1),
            ),
            transforms.RandAugment(num_ops=2, magnitude=9),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
            transforms.RandomErasing(p=0.25, scale=(0.02, 0.2)),
        ])
    else:
        return transforms.Compose([
            transforms.Resize(int(size * 1.14), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(size),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])


def make_dataloaders(config):
    root = config["data"]["root_dir"]
    paths, labels, classes, cls2idx = scan_dataset(root)
    labels_arr = np.array(labels)

    print(f"\n[DATA] {len(paths)} images | {len(classes)} classes")
    for c in classes:
        cnt = (labels_arr == cls2idx[c]).sum()
        print(f"       {c}: {cnt}")

    train_r = config["data"]["train_split"]
    val_r   = config["data"]["val_split"]
    n = len(paths)
    indices = np.arange(n)

    # First split: train vs rest
    sss1 = StratifiedShuffleSplit(n_splits=1, test_size=1 - train_r, random_state=config["seed"])
    train_idx, rest_idx = next(sss1.split(indices, labels_arr))

    # Second split: val vs test from rest
    rest_labels = labels_arr[rest_idx]
    val_ratio_of_rest = val_r / (1 - train_r)
    # Need at least 2 samples per class for stratified split — use simple split if too small
    unique, counts = np.unique(rest_labels, return_counts=True)
    if len(rest_idx) < len(unique) * 2 or (counts < 2).any():
        # Fallback: simple 50/50 split of rest
        mid = len(rest_idx) // 2
        val_rel  = np.arange(mid)
        test_rel = np.arange(mid, len(rest_idx))
    else:
        sss2 = StratifiedShuffleSplit(n_splits=1, test_size=1 - val_ratio_of_rest, random_state=config["seed"])
        val_rel, test_rel = next(sss2.split(rest_idx, rest_labels))

    val_idx  = rest_idx[val_rel]
    test_idx = rest_idx[test_rel]

    print(f"[SPLIT] Train: {len(train_idx)} | Val: {len(val_idx)} | Test: {len(test_idx)}")

    train_tf = get_transforms(config, "train")
    val_tf   = get_transforms(config, "val")

    train_ds = FractureTypeDataset(root, train_idx, paths, labels, transform=train_tf)
    val_ds   = FractureTypeDataset(root, val_idx,   paths, labels, transform=val_tf)
    test_ds  = FractureTypeDataset(root, test_idx,  paths, labels, transform=val_tf)

    # Weighted sampler for class imbalance
    train_labels = labels_arr[train_idx]
    class_counts = np.bincount(train_labels, minlength=len(classes))
    class_w = 1.0 / np.maximum(class_counts, 1)
    sample_w = class_w[train_labels]
    sampler = WeightedRandomSampler(
        weights=torch.DoubleTensor(sample_w),
        num_samples=len(train_labels),
        replacement=True,
    )

    batch = config["training"]["batch_size"]
    nw    = config["data"]["num_workers"]

    train_loader = DataLoader(train_ds, batch_size=batch, sampler=sampler,
                              num_workers=nw, pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_ds, batch_size=batch, shuffle=False,
                              num_workers=nw, pin_memory=True)
    test_loader  = DataLoader(test_ds, batch_size=batch, shuffle=False,
                              num_workers=nw, pin_memory=True)

    return train_loader, val_loader, test_loader, classes, class_counts, test_idx, labels


# ─── Model ────────────────────────────────────────────────────────────────────

class SwinFractureTypeClassifier(nn.Module):
    """Swin-Tiny + custom classification head for fracture type."""

    def __init__(self, num_classes, config):
        super().__init__()
        mc = config["model"]
        self.backbone = timm.create_model(
            mc["name"],
            pretrained=mc["pretrained"],
            num_classes=0,                      # Remove default head
            drop_path_rate=mc["drop_path_rate"],
        )
        feat_dim = self.backbone.num_features   # 768 for swin_tiny
        hc = mc["head"]
        hidden = hc["hidden_dim"]
        drop   = hc["dropout"]

        self.head = nn.Sequential(
            nn.LayerNorm(feat_dim),
            nn.Linear(feat_dim, hidden),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(hidden, num_classes),
        )
        self._init_head()

    def _init_head(self):
        for m in self.head.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        feats = self.backbone(x)
        return self.head(feats)


def build_optimizer(model, config):
    """Layer-wise LR: backbone gets 0.1x the head LR."""
    opt_cfg = config["training"]["optimizer"]
    base_lr = opt_cfg["lr"]
    backbone_lr = base_lr * 0.1

    backbone_params = list(model.backbone.parameters())
    head_params     = list(model.head.parameters())

    return torch.optim.AdamW(
        [
            {"params": backbone_params, "lr": backbone_lr},
            {"params": head_params,     "lr": base_lr},
        ],
        weight_decay=opt_cfg["weight_decay"],
        betas=tuple(opt_cfg["betas"]),
    )


# ─── Loss ─────────────────────────────────────────────────────────────────────

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, label_smoothing=0.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.label_smoothing = label_smoothing

    def forward(self, logits, targets):
        C = logits.shape[1]
        if targets.dim() == 1:
            smooth = self.label_smoothing / C
            t = torch.full_like(logits, smooth)
            if self.label_smoothing > 0:
                t.scatter_(1, targets.unsqueeze(1), 1.0 - self.label_smoothing + smooth)
            else:
                t.copy_(F.one_hot(targets, C).float())
        else:
            t = targets

        probs = F.softmax(logits, dim=-1)
        loss  = -(t * (1 - probs) ** self.gamma * F.log_softmax(logits, dim=-1))
        if self.alpha is not None:
            loss = loss * self.alpha.to(logits.device).unsqueeze(0)
        return loss.sum(dim=-1).mean()


# ─── MixUp ────────────────────────────────────────────────────────────────────

def mixup_data(x, y_onehot, alpha=0.4):
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    idx = torch.randperm(x.size(0), device=x.device)
    mixed_x = lam * x + (1 - lam) * x[idx]
    mixed_y = lam * y_onehot + (1 - lam) * y_onehot[idx]
    return mixed_x, mixed_y


# ─── Metrics ──────────────────────────────────────────────────────────────────

class AverageMeter:
    def __init__(self): self.reset()
    def reset(self): self.val = self.avg = self.sum = self.count = 0
    def update(self, val, n=1):
        self.val = val; self.sum += val * n; self.count += n; self.avg = self.sum / self.count


# ─── Training loop ────────────────────────────────────────────────────────────

def train_one_epoch(model, loader, criterion, optimizer, scaler, device, config, num_classes, epoch):
    model.train()
    loss_m = AverageMeter(); acc_m = AverageMeter()
    use_amp   = config["training"]["mixed_precision"]
    clip_norm = config["training"]["gradient_clip_norm"]
    mixup_cfg = config.get("augmentation", {}).get("train", {}).get("mixup", {})
    use_mixup = mixup_cfg.get("enabled", True)
    mixup_a   = mixup_cfg.get("alpha", 0.4)
    mixup_prob = config.get("augmentation", {}).get("train", {}).get("mixup_cutmix_prob", 0.5)

    optimizer.zero_grad()
    pbar = tqdm(loader, desc=f"  Epoch {epoch+1:3d} Train", leave=False, ncols=100)

    for images, labels in pbar:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        use_soft = use_mixup and (np.random.random() < mixup_prob)
        if use_soft:
            y_oh = F.one_hot(labels, num_classes).float()
            images, mixed_y = mixup_data(images, y_oh, mixup_a)

        use_amp_ctx = use_amp and device.type == "cuda"
        with autocast("cuda", enabled=use_amp_ctx):
            logits = model(images)
            loss = criterion(logits, mixed_y if use_soft else labels)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

        with torch.no_grad():
            preds = logits.argmax(dim=1)
            true  = (mixed_y if use_soft else F.one_hot(labels, num_classes).float()).argmax(dim=1)
            acc   = (preds == true).float().mean().item() * 100
        loss_m.update(loss.item(), images.size(0))
        acc_m.update(acc, images.size(0))
        pbar.set_postfix(loss=f"{loss_m.avg:.4f}", acc=f"{acc_m.avg:.1f}%")

    return loss_m.avg, acc_m.avg


@torch.no_grad()
def validate(model, loader, criterion, device, config, num_classes):
    model.eval()
    loss_m = AverageMeter(); acc_m = AverageMeter()
    use_amp_ctx = config["training"]["mixed_precision"] and device.type == "cuda"

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        with autocast("cuda", enabled=use_amp_ctx):
            logits = model(images)
            loss   = criterion(logits, labels)
        preds = logits.argmax(dim=1)
        acc   = (preds == labels).float().mean().item() * 100
        loss_m.update(loss.item(), images.size(0))
        acc_m.update(acc, images.size(0))

    return loss_m.avg, acc_m.avg


@torch.no_grad()
def evaluate_full(model, loader, device, classes, config):
    """Full evaluation with per-class metrics, confusion matrix, AUC."""
    model.eval()
    all_preds, all_labels, all_probs = [], [], []
    use_amp_ctx = config["training"]["mixed_precision"] and device.type == "cuda"

    for images, labels in tqdm(loader, desc="  Evaluating", leave=False):
        images = images.to(device, non_blocking=True)
        with autocast("cuda", enabled=use_amp_ctx):
            logits = model(images)
        probs = F.softmax(logits, dim=-1).cpu().numpy()
        preds = logits.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.numpy())
        all_probs.extend(probs)

    all_preds  = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs  = np.array(all_probs)

    overall_acc = (all_preds == all_labels).mean() * 100
    report = classification_report(all_labels, all_preds, target_names=classes, output_dict=True)
    macro_f1 = f1_score(all_labels, all_preds, average="macro")
    cm = confusion_matrix(all_labels, all_preds)

    try:
        auc = roc_auc_score(
            np.eye(len(classes))[all_labels],
            all_probs,
            multi_class="ovr", average="macro"
        )
    except Exception:
        auc = float("nan")

    return overall_acc, report, macro_f1, auc, cm


# ─── CSV export ───────────────────────────────────────────────────────────────

def save_results_csv(history, overall_acc, report, macro_f1, auc_roc, cm, classes, config, training_time_s):
    os.makedirs(config["paths"]["results_dir"], exist_ok=True)

    # --- Per-epoch performance CSV ---
    rows = []
    for i in range(len(history["train_loss"])):
        gap = history["val_loss"][i] - history["train_loss"][i]
        rows.append({
            "epoch": i + 1,
            "train_loss": round(history["train_loss"][i], 6),
            "val_loss":   round(history["val_loss"][i], 6),
            "train_accuracy": round(history["train_acc"][i], 2),
            "val_accuracy":   round(history["val_acc"][i], 2),
            "overfitting_gap": round(gap, 6),
            "learning_rate":  f"{history['lr'][i]:.2e}",
        })
    df_perf = pd.DataFrame(rows)
    perf_path = os.path.join(config["paths"]["results_dir"], "model_performance_analysis.csv")
    df_perf.to_csv(perf_path, index=False)
    best_val = max(history["val_acc"])
    best_ep  = history["val_acc"].index(best_val) + 1
    max_gap  = max(history["val_loss"][i] - history["train_loss"][i] for i in range(len(history["train_loss"])))
    with open(perf_path, "a") as f:
        f.write(f"\n# GENERALIZATION METRICS (Swin-Tiny Fracture Type):\n")
        f.write(f"# - Max Overfitting Gap: {max_gap:.4f}\n")
        f.write(f"# - Best Val Accuracy: {best_val:.2f}% (epoch {best_ep})\n")
        f.write(f"# - Test Accuracy: {overall_acc:.2f}%\n")
        f.write(f"# - Train/Test Accuracy Delta: {history['train_acc'][-1] - overall_acc:.2f}%\n")
        f.write(f"# - Macro F1: {macro_f1:.4f} | AUC-ROC: {auc_roc:.4f}\n")
        f.write(f"# - Training Time: {training_time_s/60:.1f} minutes\n")
    print(f"[SAVE] Performance analysis -> {perf_path}")

    # --- Final results CSV ---
    results_rows = []
    results_rows.append({"metric_name": "Accuracy", "overall_value": f"{overall_acc:.3f}",
                          **{f"class_{c}_value": "N/A" for c in classes},
                          "interpretation": "Overall test set correctness"})
    results_rows.append({"metric_name": "Macro F1-Score", "overall_value": f"{macro_f1:.3f}",
                          **{f"class_{c}_value": f"{report[c]['f1-score']:.3f}" for c in classes},
                          "interpretation": "Balanced per-class F1"})
    results_rows.append({"metric_name": "Precision", "overall_value": f"{report['macro avg']['precision']:.3f}",
                          **{f"class_{c}_value": f"{report[c]['precision']:.3f}" for c in classes},
                          "interpretation": "Positive prediction reliability"})
    results_rows.append({"metric_name": "Recall", "overall_value": f"{report['macro avg']['recall']:.3f}",
                          **{f"class_{c}_value": f"{report[c]['recall']:.3f}" for c in classes},
                          "interpretation": "Detection rate per class"})
    results_rows.append({"metric_name": "AUC-ROC", "overall_value": f"{auc_roc:.3f}",
                          **{f"class_{c}_value": "N/A" for c in classes},
                          "interpretation": "Threshold-independent discriminability"})
    results_rows.append({"metric_name": "Training_Time_min", "overall_value": f"{training_time_s/60:.1f}",
                          **{f"class_{c}_value": "N/A" for c in classes},
                          "interpretation": "Total training time in minutes"})

    df_res = pd.DataFrame(results_rows)
    res_path = os.path.join(config["paths"]["results_dir"], "final_results.csv")
    df_res.to_csv(res_path, index=False)
    print(f"[SAVE] Final results -> {res_path}")

    # --- Confusion matrix ---
    cm_df = pd.DataFrame(cm, index=classes, columns=classes)
    cm_path = os.path.join(config["paths"]["results_dir"], "confusion_matrix.csv")
    cm_df.to_csv(cm_path)
    print(f"[SAVE] Confusion matrix -> {cm_path}")


# ─── Main training function ───────────────────────────────────────────────────

def train(config):
    seed_everything(config["seed"])
    device = get_device()
    os.makedirs(config["paths"]["checkpoint_dir"], exist_ok=True)
    os.makedirs(config["paths"]["results_dir"], exist_ok=True)

    # Data
    print("\n" + "="*70)
    print("  LOADING BONE BREAK CLASSIFICATION DATASET")
    print("="*70)
    train_loader, val_loader, test_loader, classes, class_counts, test_idx, all_labels = \
        make_dataloaders(config)
    num_classes = len(classes)

    # Save class names for inference
    cn_path = os.path.join(config["paths"]["checkpoint_dir"], "class_names.json")
    with open(cn_path, "w") as f:
        json.dump(classes, f, indent=2)
    print(f"[SAVE] Class names -> {cn_path}")

    # Model
    print("\n" + "="*70)
    print("  BUILDING SWIN-TINY FRACTURE TYPE CLASSIFIER")
    print("="*70)
    model = SwinFractureTypeClassifier(num_classes, config).to(device)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    n_train  = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    print(f"  Parameters: {n_params:.1f}M total, {n_train:.1f}M trainable")

    # Loss
    class_w = torch.FloatTensor(1.0 / np.maximum(class_counts, 1))
    class_w = (class_w / class_w.sum() * num_classes).to(device)
    criterion = FocalLoss(alpha=class_w, gamma=config["training"]["focal_loss"]["gamma"],
                          label_smoothing=config["training"]["label_smoothing"])

    # Optimizer (layer-wise LR)
    optimizer = build_optimizer(model, config)

    # Scheduler
    sc = config["training"]["scheduler"]
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=sc["T_0"], T_mult=sc["T_mult"], eta_min=sc["eta_min"]
    )

    scaler = GradScaler(enabled=config["training"]["mixed_precision"] and device.type == "cuda")

    # Early stopping
    es_cfg = config["training"]["early_stopping"]
    patience = es_cfg["patience"]
    min_delta = es_cfg.get("min_delta", 0.1)
    best_val_acc = 0.0
    no_improve   = 0
    best_epoch   = 0
    ckpt_path    = os.path.join(config["paths"]["checkpoint_dir"], "best.pth")

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": [], "lr": []}

    # Warmup
    warmup_cfg = config["training"]["warmup"]
    warmup_ep  = warmup_cfg["epochs"]
    warmup_lr  = warmup_cfg["start_lr"]
    target_lr  = config["training"]["optimizer"]["lr"]

    num_epochs = config["training"]["epochs"]
    t_start = time.time()

    print(f"\n{'='*70}")
    print(f"  TRAINING — {num_epochs} epochs | {num_classes} classes | batch {config['training']['batch_size']}")
    print(f"{'='*70}\n")

    for epoch in range(num_epochs):
        # Warmup LR
        if epoch < warmup_ep:
            warmup_lrv = warmup_lr + (target_lr - warmup_lr) * (epoch / max(warmup_ep, 1))
            for pg in optimizer.param_groups:
                pg["lr"] = warmup_lrv * (0.1 if pg is optimizer.param_groups[0] else 1.0)

        current_lr = optimizer.param_groups[1]["lr"]  # head LR

        ep_t = time.time()
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, device, config, num_classes, epoch
        )
        val_loss, val_acc = validate(model, val_loader, criterion, device, config, num_classes)

        if epoch >= warmup_ep:
            scheduler.step()

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        history["lr"].append(current_lr)

        gap = val_loss - train_loss
        ep_time = time.time() - ep_t

        print(f"Epoch {epoch+1:3d}/{num_epochs} | "
              f"Train {train_loss:.4f}/{train_acc:.1f}% | "
              f"Val {val_loss:.4f}/{val_acc:.1f}% | "
              f"Gap {gap:+.4f} | LR {current_lr:.2e} | {ep_time:.1f}s")

        # Save best
        if val_acc > best_val_acc + min_delta:
            best_val_acc = val_acc
            best_epoch   = epoch + 1
            no_improve   = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_acc": val_acc,
                "class_names": classes,
                "config": config,
            }, ckpt_path)
            print(f"  ★ New best: {val_acc:.2f}% → saved to {ckpt_path}")
        else:
            no_improve += 1

        if no_improve >= patience:
            print(f"\n  [EARLY STOP] No improvement for {patience} epochs.")
            break

    training_time = time.time() - t_start
    print(f"\n{'='*70}")
    print(f"  TRAINING COMPLETE | Best Val: {best_val_acc:.2f}% (Epoch {best_epoch})")
    print(f"  Total time: {training_time/60:.1f} minutes")
    print(f"{'='*70}")

    # Full test evaluation
    print("\n[EVAL] Loading best checkpoint for test evaluation...")
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    overall_acc, report, macro_f1, auc_roc, cm = evaluate_full(model, test_loader, device, classes, config)

    print(f"\n[RESULTS] Test Accuracy: {overall_acc:.2f}%")
    print(f"          Macro F1-Score: {macro_f1:.4f}")
    print(f"          AUC-ROC (macro ovr): {auc_roc:.4f}")
    print("\n[Per-class metrics]")
    for cls in classes:
        r = report[cls]
        print(f"  {cls:<25} P={r['precision']:.3f}  R={r['recall']:.3f}  F1={r['f1-score']:.3f}  N={int(r['support'])}")

    save_results_csv(history, overall_acc, report, macro_f1, auc_roc, cm, classes, config, training_time)

    return best_val_acc, overall_acc, macro_f1


# ─── Eval-only mode ───────────────────────────────────────────────────────────

def eval_only(config):
    device  = get_device()
    ckpt_path = os.path.join(config["paths"]["checkpoint_dir"], "best.pth")
    if not os.path.exists(ckpt_path):
        print(f"[ERROR] No checkpoint found at {ckpt_path}")
        return

    ckpt = torch.load(ckpt_path, map_location=device)
    classes = ckpt["class_names"]
    num_classes = len(classes)
    config["data"]["num_classes"] = num_classes

    model = SwinFractureTypeClassifier(num_classes, config).to(device)
    model.load_state_dict(ckpt["model_state_dict"])

    _, val_loader, test_loader, _, _, _, _ = make_dataloaders(config)
    acc, report, macro_f1, auc_roc, cm = evaluate_full(model, test_loader, device, classes, config)
    print(f"\n[TEST] Accuracy: {acc:.2f}% | Macro F1: {macro_f1:.4f} | AUC: {auc_roc:.4f}")
    for cls in classes:
        r = report[cls]
        print(f"  {cls:<25} F1={r['f1-score']:.3f}")


# ─── Entry point ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train Swin-Tiny Fracture Type Classifier")
    parser.add_argument("--config",    type=str, default="configs/fracture_type_config.yaml")
    parser.add_argument("--epochs",    type=int, default=None, help="Override epochs")
    parser.add_argument("--eval-only", action="store_true",    help="Just evaluate saved model")
    parser.add_argument("--debug",     action="store_true",    help="2 epochs, batch=4")
    args = parser.parse_args()

    if os.path.exists(args.config):
        config = load_config(args.config)
    else:
        print(f"[WARN] Config not found at {args.config}, using defaults.")
        config = DEFAULT_CONFIG

    if args.epochs:
        config["training"]["epochs"] = args.epochs
    if args.debug:
        config["training"]["epochs"] = 2
        config["training"]["batch_size"] = 4
        config["data"]["num_workers"] = 0
        print("[DEBUG] 2 epochs, batch=4, num_workers=0")

    if args.eval_only:
        eval_only(config)
    else:
        train(config)


if __name__ == "__main__":
    main()
