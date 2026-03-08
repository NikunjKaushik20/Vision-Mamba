"""
========================================================================
FractureMamba-ViT — Google Colab Training Script
All-in-one script: data loading, model, training, evaluation
========================================================================

INSTRUCTIONS (paste these into Colab cells):

CELL 1 — Setup:
    !pip install timm einops albumentations pyyaml -q
    from google.colab import drive
    drive.mount('/content/drive')

CELL 2 — Upload dataset:
    # Option A: Upload from Kaggle directly
    !pip install kaggle -q
    # Upload your kaggle.json to Colab, then:
    !mkdir -p ~/.kaggle && cp kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json
    !kaggle datasets download -d prathamsharma123/bone-fracture-multi-region-x-ray-data
    !unzip -q bone-fracture-multi-region-x-ray-data.zip -d /content/dataset

    # Option B: If dataset is already on Google Drive
    !cp -r "/content/drive/MyDrive/Bone_Fracture_Binary_Classification" /content/dataset/

CELL 3 — Upload this script:
    # Upload colab_train.py to Colab, then:
    !python colab_train.py --data-dir /content/dataset/Bone_Fracture_Binary_Classification --epochs 50

CELL 4 — Copy results back to Drive:
    !cp -r /content/results "/content/drive/MyDrive/FractureMambaViT_Results/"
    !cp -r /content/checkpoints "/content/drive/MyDrive/FractureMambaViT_Results/"
"""

import os
import sys
import time
import math
import random
import argparse
import numpy as np
import pandas as pd
import cv2
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
from torch.optim.swa_utils import AveragedModel, SWALR
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score,
)
from PIL import Image
from pathlib import Path
from einops import rearrange
import timm
import albumentations as A
from albumentations.pytorch import ToTensorV2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# =====================================================================
# CONFIGURATION
# =====================================================================
DEFAULT_CONFIG = {
    "data": {
        "image_size": 224,
        "num_classes": 2,
        "num_workers": 2,
        "pin_memory": True,
    },
    "model": {
        "mamba": {
            "embed_dim": 192, "depth": 4, "patch_size": 16,
            "d_state": 8, "d_conv": 4, "expand_factor": 2,
            "drop_path_rate": 0.2, "dropout": 0.1,
        },
        "swin": {
            "model_name": "swin_tiny_patch4_window7_224",
            "pretrained": True, "drop_path_rate": 0.2,
        },
        "fusion": {"dim": 384, "num_heads": 8, "dropout": 0.1},
        "head": {"hidden_dim": 512, "dropout": 0.3},
    },
    "training": {
        "epochs": 50,
        "batch_size": 16,  # Colab T4 has 16GB, can go bigger
        "gradient_accumulation_steps": 2,
        "optimizer": {"name": "AdamW", "lr": 1e-4, "weight_decay": 0.05, "betas": [0.9, 0.999]},
        "scheduler": {"name": "CosineAnnealingWarmRestarts", "T_0": 10, "T_mult": 2, "eta_min": 1e-6},
        "warmup": {"epochs": 5, "start_lr": 1e-6},
        "focal_loss": {"gamma": 2.0, "alpha": None},
        "label_smoothing": 0.1,
        "gradient_clip_norm": 1.0,
        "early_stopping": {"patience": 20, "min_delta": 0.001},
        "swa": {"enabled": True, "start_epoch": 40, "lr": 5e-5},
        "mixed_precision": True,
    },
    "augmentation": {
        "train": {
            "clahe": {"enabled": True, "clip_limit": 2.0, "tile_grid_size": 8},
            "horizontal_flip": 0.5, "rotation": 15,
            "affine": {"translate": [0.1, 0.1], "scale": [0.9, 1.1], "shear": 5},
            "color_jitter": {"brightness": 0.2, "contrast": 0.2, "saturation": 0.1, "hue": 0.05},
            "random_erasing": {"probability": 0.25},
            "gaussian_blur": {"probability": 0.1, "kernel_size": 3},
            "mixup": {"enabled": True, "alpha": 0.4},
            "cutmix": {"enabled": True, "alpha": 1.0},
            "mixup_cutmix_prob": 0.5,
        },
        "normalize": {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]},
    },
    "evaluation": {"tta": {"enabled": True, "num_augments": 5}},
    "paths": {
        "checkpoint_dir": "/content/checkpoints",
        "results_dir": "/content/results",
        "explainability_dir": "/content/explainability_outputs",
        "logs_dir": "/content/logs",
    },
    "seed": 42,
    "deterministic": True,
}


# =====================================================================
# UTILITIES
# =====================================================================
def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"[INFO] GPU: {torch.cuda.get_device_name(0)}")
        print(f"[INFO] VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    else:
        device = torch.device("cpu")
        print("[WARN] No GPU! Training will be very slow.")
    return device

class AverageMeter:
    def __init__(self):
        self.reset()
    def reset(self):
        self.avg = 0; self.sum = 0; self.count = 0
    def update(self, val, n=1):
        self.sum += val * n; self.count += n; self.avg = self.sum / self.count

class EarlyStopping:
    def __init__(self, patience=15):
        self.patience = patience; self.counter = 0; self.best_score = None
    def __call__(self, score):
        if self.best_score is None or score > self.best_score + 0.001:
            self.best_score = score; self.counter = 0; return False
        self.counter += 1
        return self.counter >= self.patience


# =====================================================================
# DATA LOADING
# =====================================================================
class BoneFractureDataset(Dataset):
    def __init__(self, root_dir, split="train", transform=None):
        self.root_dir = Path(root_dir) / split
        self.transform = transform
        self.classes = sorted([d for d in os.listdir(self.root_dir) if os.path.isdir(self.root_dir / d)])
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        self.images, self.labels = [], []
        for cls in self.classes:
            cls_dir = self.root_dir / cls
            for img_name in os.listdir(cls_dir):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                    self.images.append(str(cls_dir / img_name))
                    self.labels.append(self.class_to_idx[cls])
        self.labels = np.array(self.labels)
        class_counts = np.bincount(self.labels, minlength=len(self.classes))
        cw = 1.0 / (class_counts + 1e-6)
        self.sample_weights = (cw / cw.sum())[self.labels]
        print(f"[DATA] {split}: {len(self.images)} images, classes: {self.classes}")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = cv2.imread(self.images[idx])
        if image is None:
            image = np.array(Image.open(self.images[idx]).convert("RGB"))
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.transform:
            image = self.transform(image=image)["image"]
        return image, torch.tensor(self.labels[idx], dtype=torch.long)


def get_train_transforms(img_size=224):
    return A.Compose([
        A.Resize(img_size, img_size),
        A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=1.0),
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=15, p=0.5, border_mode=cv2.BORDER_REFLECT_101),
        A.Affine(translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)},
                 scale=(0.9, 1.1), rotate=(-5, 5), p=0.4, border_mode=cv2.BORDER_REFLECT_101),
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05, p=0.4),
        A.GaussNoise(p=0.2),
        A.GridDistortion(num_steps=5, distort_limit=0.1, p=0.2),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])

def get_val_transforms(img_size=224):
    return A.Compose([
        A.Resize(img_size, img_size),
        A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=1.0),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])


class MixupCutmix:
    def __init__(self, alpha_mix=0.4, alpha_cut=1.0, prob=0.5, num_classes=2):
        self.alpha_mix = alpha_mix; self.alpha_cut = alpha_cut
        self.prob = prob; self.nc = num_classes

    def __call__(self, images, labels):
        B = images.size(0)
        oh = torch.zeros(B, self.nc, device=images.device).scatter_(1, labels.unsqueeze(1), 1.0)
        if np.random.random() > self.prob:
            return images, oh
        idx = torch.randperm(B, device=images.device)
        if np.random.random() < 0.5:  # Mixup
            lam = max(np.random.beta(self.alpha_mix, self.alpha_mix), 0.5)
            return lam * images + (1-lam) * images[idx], lam * oh + (1-lam) * oh[idx]
        else:  # CutMix
            lam = np.random.beta(self.alpha_cut, self.alpha_cut)
            _, _, H, W = images.shape
            rh, rw = int(H * np.sqrt(1-lam)), int(W * np.sqrt(1-lam))
            cy, cx = np.random.randint(H), np.random.randint(W)
            y1, y2 = np.clip(cy-rh//2, 0, H), np.clip(cy+rh//2, 0, H)
            x1, x2 = np.clip(cx-rw//2, 0, W), np.clip(cx+rw//2, 0, W)
            mixed = images.clone()
            mixed[:, :, y1:y2, x1:x2] = images[idx, :, y1:y2, x1:x2]
            lam = 1 - (y2-y1)*(x2-x1)/(H*W)
            return mixed, lam * oh + (1-lam) * oh[idx]


# =====================================================================
# VISION MAMBA MODULE (Pure PyTorch — no CUDA kernels)
# =====================================================================
class PatchEmbedding(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_ch=3, embed_dim=192):
        super().__init__()
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_ch, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)
    def forward(self, x):
        x = self.proj(x)
        x = rearrange(x, "b c h w -> b (h w) c")
        return self.norm(x)


class SelectiveSSM(nn.Module):
    def __init__(self, d_model, d_state=8, d_conv=4, expand_factor=2, dropout=0.1):
        super().__init__()
        self.d_state = d_state
        self.d_inner = int(d_model * expand_factor)
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)
        self.conv1d = nn.Conv1d(self.d_inner, self.d_inner, kernel_size=d_conv,
                                padding=d_conv-1, groups=self.d_inner, bias=True)
        self.x_proj = nn.Linear(self.d_inner, d_state * 2 + 1, bias=False)
        A = torch.arange(1, d_state+1, dtype=torch.float32).unsqueeze(0).expand(self.d_inner, -1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.dt_proj = nn.Linear(1, self.d_inner, bias=True)
        dt = torch.exp(torch.rand(self.d_inner) * (math.log(0.1)-math.log(0.001)) + math.log(0.001))
        with torch.no_grad():
            self.dt_proj.bias.copy_(dt + torch.log(-torch.expm1(-dt)))
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def selective_scan(self, u, delta, A, B, C, D):
        batch_size, seq_len, d_inner = u.shape
        d_state = A.shape[1]
        CHUNK = 32
        delta_A = torch.exp(delta.unsqueeze(-1) * A.unsqueeze(0).unsqueeze(0)).clamp(max=1.0)
        delta_B_u = (delta.unsqueeze(-1) * B.unsqueeze(2)) * u.unsqueeze(-1)
        ys = []
        h = torch.zeros(batch_size, d_inner, d_state, device=u.device, dtype=u.dtype)
        for start in range(0, seq_len, CHUNK):
            end = min(start + CHUNK, seq_len)
            dA_c = delta_A[:, start:end]
            dBu_c = delta_B_u[:, start:end]
            C_c = C[:, start:end]
            log_dA = torch.log(dA_c.clamp(min=1e-8))
            cum_log = torch.cumsum(log_dA, dim=1)
            cum_dA = torch.exp(cum_log.clamp(max=20.0))
            inv_cum = torch.exp(-cum_log.clamp(min=-20.0, max=20.0))
            states = cum_dA * (h.unsqueeze(1) + torch.cumsum(dBu_c * inv_cum, dim=1))
            ys.append((states * C_c.unsqueeze(2)).sum(-1))
            h = states[:, -1]
        y = torch.cat(ys, dim=1) + u * D.unsqueeze(0).unsqueeze(0)
        return y

    def forward(self, x):
        B, L, _ = x.shape
        xz = self.in_proj(x)
        x_b, z = xz.chunk(2, dim=-1)
        x_c = F.silu(rearrange(self.conv1d(rearrange(x_b, "b l d -> b d l"))[:, :, :L], "b d l -> b l d"))
        ssm = self.x_proj(x_c)
        B_p, C_p, dt_p = ssm[:,:,:self.d_state], ssm[:,:,self.d_state:2*self.d_state], ssm[:,:,-1:]
        dt = F.softplus(self.dt_proj(dt_p))
        A = -torch.exp(self.A_log)
        y = self.selective_scan(x_c, dt, A, B_p, C_p, self.D)
        return self.dropout(self.out_proj(y * F.silu(z)))


class DropPath(nn.Module):
    def __init__(self, p=0.0):
        super().__init__(); self.p = p
    def forward(self, x):
        if not self.training or self.p == 0: return x
        keep = 1 - self.p
        mask = torch.floor(torch.rand((x.shape[0],)+(1,)*(x.ndim-1), device=x.device) + keep)
        return x / keep * mask

class MambaBlock(nn.Module):
    def __init__(self, d, d_state=8, d_conv=4, expand=2, drop=0.1, dp=0.0):
        super().__init__()
        self.norm = nn.LayerNorm(d)
        self.ssm = SelectiveSSM(d, d_state, d_conv, expand, drop)
        self.dp = DropPath(dp) if dp > 0 else nn.Identity()
    def forward(self, x):
        return x + self.dp(self.ssm(self.norm(x)))

class BidirectionalMambaBlock(nn.Module):
    def __init__(self, d, d_state=8, d_conv=4, expand=2, drop=0.1, dp=0.0):
        super().__init__()
        self.fwd = MambaBlock(d, d_state, d_conv, expand, drop, dp)
        self.bwd = MambaBlock(d, d_state, d_conv, expand, drop, dp)
        self.fuse = nn.Linear(d*2, d)
        self.norm = nn.LayerNorm(d)
    def forward(self, x):
        return self.norm(self.fuse(torch.cat([self.fwd(x), self.bwd(x.flip(1)).flip(1)], dim=-1)))

class VisionMamba(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_ch=3, embed_dim=192,
                 depth=4, d_state=8, d_conv=4, expand=2, drop=0.1, drop_path=0.2):
        super().__init__()
        self.embed_dim = embed_dim
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_ch, embed_dim)
        n = self.patch_embed.num_patches
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, n+1, embed_dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.pos_drop = nn.Dropout(drop)
        dpr = [x.item() for x in torch.linspace(0, drop_path, depth)]
        self.blocks = nn.ModuleList([
            BidirectionalMambaBlock(embed_dim, d_state, d_conv, expand, drop, dpr[i])
            for i in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)
        x = torch.cat([self.cls_token.expand(B,-1,-1), x], dim=1) + self.pos_embed
        x = self.pos_drop(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return x[:, 0], x[:, 1:]


# =====================================================================
# MAIN MODEL: FractureMamba-ViT
# =====================================================================
class CrossAttentionFusion(nn.Module):
    def __init__(self, dim, num_heads=8, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        self.q_m = nn.Linear(dim, dim); self.k_s = nn.Linear(dim, dim); self.v_s = nn.Linear(dim, dim)
        self.q_s = nn.Linear(dim, dim); self.k_m = nn.Linear(dim, dim); self.v_m = nn.Linear(dim, dim)
        self.attn_drop = nn.Dropout(dropout)
        self.gate = nn.Sequential(nn.Linear(dim*2, dim), nn.Sigmoid())
        self.out_proj = nn.Linear(dim, dim)
        self.norm1 = nn.LayerNorm(dim); self.norm2 = nn.LayerNorm(dim); self.norm3 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(nn.Linear(dim, dim*4), nn.GELU(), nn.Dropout(dropout),
                                  nn.Linear(dim*4, dim), nn.Dropout(dropout))
        self.dropout = nn.Dropout(dropout)

    def _xattn(self, q, k, v):
        B, N, _ = q.shape
        q = rearrange(q, "b n (h d) -> b h n d", h=self.num_heads)
        k = rearrange(k, "b m (h d) -> b h m d", h=self.num_heads)
        v = rearrange(v, "b m (h d) -> b h m d", h=self.num_heads)
        attn = F.softmax(torch.matmul(q, k.transpose(-2,-1)) * self.scale, dim=-1)
        return rearrange(torch.matmul(self.attn_drop(attn), v), "b h n d -> b n (h d)")

    def forward(self, mamba_f, swin_f):
        ms = mamba_f.unsqueeze(1); ss = swin_f.unsqueeze(1)
        m2s = self.norm1(ms + self._xattn(self.q_m(ms), self.k_s(ss), self.v_s(ss))).squeeze(1)
        s2m = self.norm2(ss + self._xattn(self.q_s(ss), self.k_m(ms), self.v_m(ms))).squeeze(1)
        g = self.gate(torch.cat([m2s, s2m], dim=-1))
        fused = (g * m2s + (1-g) * s2m).unsqueeze(1)
        fused = (fused + self.ffn(self.norm3(fused))).squeeze(1)
        return self.dropout(self.out_proj(fused))


class FractureMambaViT(nn.Module):
    def __init__(self, num_classes=2, config=None):
        super().__init__()
        cfg = (config or DEFAULT_CONFIG)["model"]
        mc, sc, fc, hc = cfg["mamba"], cfg["swin"], cfg["fusion"], cfg["head"]
        img_size = (config or DEFAULT_CONFIG).get("data", {}).get("image_size", 224)

        self.mamba_stream = VisionMamba(
            img_size=img_size, patch_size=mc["patch_size"], embed_dim=mc["embed_dim"],
            depth=mc["depth"], d_state=mc["d_state"], d_conv=mc["d_conv"],
            expand=mc["expand_factor"], drop=mc["dropout"], drop_path=mc["drop_path_rate"])

        self.swin_stream = timm.create_model(sc["model_name"], pretrained=sc["pretrained"],
                                              num_classes=0, drop_path_rate=sc["drop_path_rate"])
        swin_dim = self.swin_stream.num_features

        self.mamba_proj = nn.Sequential(nn.Linear(mc["embed_dim"], fc["dim"]), nn.LayerNorm(fc["dim"]), nn.GELU())
        self.swin_proj = nn.Sequential(nn.Linear(swin_dim, fc["dim"]), nn.LayerNorm(fc["dim"]), nn.GELU())
        self.fusion = CrossAttentionFusion(fc["dim"], fc["num_heads"], fc["dropout"])
        self.classifier = nn.Sequential(
            nn.Linear(fc["dim"], hc["hidden_dim"]), nn.LayerNorm(hc["hidden_dim"]), nn.GELU(), nn.Dropout(hc["dropout"]),
            nn.Linear(hc["hidden_dim"], hc["hidden_dim"]//2), nn.GELU(), nn.Dropout(hc["dropout"]*0.5),
            nn.Linear(hc["hidden_dim"]//2, num_classes))

    def forward(self, x):
        m_cls, _ = self.mamba_stream(x)
        s_feat = self.swin_stream(x)
        fused = self.fusion(self.mamba_proj(m_cls), self.swin_proj(s_feat))
        return self.classifier(fused)


class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, label_smoothing=0.0):
        super().__init__()
        self.alpha = alpha; self.gamma = gamma; self.ls = label_smoothing
    def forward(self, logits, targets):
        nc = logits.shape[1]
        if targets.dim() == 1:
            smooth = self.ls / nc
            t = torch.full_like(logits, smooth)
            t.scatter_(1, targets.unsqueeze(1), 1.0 - self.ls + smooth)
        else:
            t = targets
        p = F.softmax(logits, dim=-1)
        loss = -t * ((1-p)**self.gamma) * F.log_softmax(logits, dim=-1)
        if self.alpha is not None:
            loss = loss * self.alpha.to(logits.device).unsqueeze(0)
        return loss.sum(dim=-1).mean()


# =====================================================================
# TRAINING
# =====================================================================
def train_one_epoch(model, loader, criterion, optimizer, scaler, device, config, mixup_fn=None, epoch=0):
    model.train()
    loss_m, acc_m = AverageMeter(), AverageMeter()
    accum = config["training"]["gradient_accumulation_steps"]
    optimizer.zero_grad()
    pbar = tqdm(loader, desc=f"  Train {epoch+1}", leave=False, ncols=100)
    for step, (imgs, labels) in enumerate(pbar):
        imgs, labels = imgs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        soft = False
        if mixup_fn and np.random.random() < 0.5:
            imgs, labels_m = mixup_fn(imgs, labels); soft = True
        with autocast('cuda', enabled=config["training"]["mixed_precision"]):
            logits = model(imgs)
            loss = criterion(logits, labels_m if soft else labels) / accum
        scaler.scale(loss).backward()
        if (step+1) % accum == 0 or (step+1) == len(loader):
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config["training"]["gradient_clip_norm"])
            scaler.step(optimizer); scaler.update(); optimizer.zero_grad()
        with torch.no_grad():
            pred = logits.argmax(1)
            true = (labels_m.argmax(1) if soft else labels)
            acc_m.update((pred == true).float().mean().item() * 100, imgs.size(0))
        loss_m.update(loss.item() * accum, imgs.size(0))
        pbar.set_postfix(loss=f"{loss_m.avg:.4f}", acc=f"{acc_m.avg:.1f}%")
    return loss_m.avg, acc_m.avg

@torch.no_grad()
def validate(model, loader, criterion, device, use_amp=True):
    model.eval()
    loss_m, acc_m = AverageMeter(), AverageMeter()
    for imgs, labels in loader:
        imgs, labels = imgs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        with autocast('cuda', enabled=use_amp):
            logits = model(imgs); loss = criterion(logits, labels)
        acc_m.update((logits.argmax(1) == labels).float().mean().item() * 100, imgs.size(0))
        loss_m.update(loss.item(), imgs.size(0))
    return loss_m.avg, acc_m.avg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, required=True, help="Path to dataset root (containing train/val/test)")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    config = DEFAULT_CONFIG.copy()
    config["training"]["epochs"] = 2 if args.debug else args.epochs
    config["training"]["batch_size"] = 4 if args.debug else args.batch_size
    config["training"]["optimizer"]["lr"] = args.lr

    seed_everything(42)
    device = get_device()
    for k in ["checkpoint_dir", "results_dir", "explainability_dir", "logs_dir"]:
        os.makedirs(config["paths"][k], exist_ok=True)

    # Data
    img_size = config["data"]["image_size"]
    train_ds = BoneFractureDataset(args.data_dir, "train", get_train_transforms(img_size))
    val_ds = BoneFractureDataset(args.data_dir, "val", get_val_transforms(img_size))
    test_ds = BoneFractureDataset(args.data_dir, "test", get_val_transforms(img_size))

    num_classes = len(train_ds.classes)
    config["data"]["num_classes"] = num_classes
    bs = config["training"]["batch_size"]
    nw = config["data"]["num_workers"]

    sampler = WeightedRandomSampler(torch.DoubleTensor(train_ds.sample_weights), len(train_ds.sample_weights), replacement=True)
    train_loader = DataLoader(train_ds, batch_size=bs, sampler=sampler, num_workers=nw, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False, num_workers=nw, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=bs, shuffle=False, num_workers=nw, pin_memory=True)

    # Model
    model = FractureMambaViT(num_classes=num_classes, config=config).to(device)
    params = sum(p.numel() for p in model.parameters())
    print(f"[MODEL] FractureMamba-ViT: {params:,} parameters")

    # Loss
    cc = np.bincount(train_ds.labels, minlength=num_classes)
    alpha = torch.FloatTensor(cc.sum() / (num_classes * cc))
    criterion = FocalLoss(alpha=alpha, gamma=2.0, label_smoothing=0.1)

    # Optimizer + Scheduler
    opt_cfg = config["training"]["optimizer"]
    optimizer = torch.optim.AdamW(model.parameters(), lr=opt_cfg["lr"],
                                   weight_decay=opt_cfg["weight_decay"], betas=tuple(opt_cfg["betas"]))
    sched_cfg = config["training"]["scheduler"]
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=sched_cfg["T_0"], T_mult=sched_cfg["T_mult"], eta_min=sched_cfg["eta_min"])
    scaler = GradScaler('cuda', enabled=config["training"]["mixed_precision"])

    mixup_fn = MixupCutmix(num_classes=num_classes)
    swa_cfg = config["training"]["swa"]
    swa_model = AveragedModel(model) if swa_cfg["enabled"] else None
    swa_sched = SWALR(optimizer, swa_lr=swa_cfg["lr"], anneal_epochs=5) if swa_cfg["enabled"] else None
    early_stop = EarlyStopping(patience=config["training"]["early_stopping"]["patience"])

    warmup_epochs = config["training"]["warmup"]["epochs"]
    warmup_start = config["training"]["warmup"]["start_lr"]
    target_lr = opt_cfg["lr"]
    num_epochs = config["training"]["epochs"]
    swa_start = swa_cfg.get("start_epoch", 40)
    best_val_acc, best_epoch = 0.0, 0
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": [], "lr": []}

    print(f"\n{'='*60}")
    print(f"  TRAINING — {num_epochs} epochs, batch {bs}×{config['training']['gradient_accumulation_steps']}")
    print(f"{'='*60}\n")

    t0 = time.time()
    for epoch in range(num_epochs):
        if epoch < warmup_epochs:
            lr = warmup_start + (target_lr - warmup_start) * (epoch / warmup_epochs)
            for pg in optimizer.param_groups: pg["lr"] = lr
        cur_lr = optimizer.param_groups[0]["lr"]

        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, scaler, device, config, mixup_fn, epoch)
        val_loss, val_acc = validate(model, val_loader, criterion, device, config["training"]["mixed_precision"])

        if epoch >= warmup_epochs:
            if swa_model and epoch >= swa_start:
                swa_model.update_parameters(model); swa_sched.step()
            else:
                scheduler.step()

        history["train_loss"].append(train_loss); history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc); history["val_acc"].append(val_acc)
        history["lr"].append(cur_lr)

        gap = val_loss - train_loss
        is_best = val_acc > best_val_acc
        if is_best: best_val_acc = val_acc; best_epoch = epoch + 1

        print(f"Epoch {epoch+1:3d}/{num_epochs} | TrL: {train_loss:.4f} TrA: {train_acc:.1f}% | "
              f"VaL: {val_loss:.4f} VaA: {val_acc:.1f}% | Gap: {gap:.4f} | LR: {cur_lr:.2e}")

        ckpt = {"epoch": epoch, "model_state_dict": model.state_dict(), "val_acc": val_acc}
        torch.save(ckpt, os.path.join(config["paths"]["checkpoint_dir"], "checkpoint.pth"))
        if is_best:
            torch.save(ckpt, os.path.join(config["paths"]["checkpoint_dir"], "checkpoint_best.pth"))
            print(f"  [BEST] val_acc={val_acc:.2f}%")

        if early_stop(val_acc):
            print(f"\n  [EARLY STOP] No improvement for {config['training']['early_stopping']['patience']} epochs.")
            break

    total_time = time.time() - t0

    if swa_model:
        print("\n[SWA] Updating batch norm...")
        torch.optim.swa_utils.update_bn(train_loader, swa_model, device=device)
        torch.save({"model_state_dict": swa_model.module.state_dict()},
                    os.path.join(config["paths"]["checkpoint_dir"], "checkpoint_swa.pth"))

    # --- Evaluate on test set ---
    print(f"\n{'='*60}\n  EVALUATING ON TEST SET\n{'='*60}")
    best_ckpt = os.path.join(config["paths"]["checkpoint_dir"], "checkpoint_best.pth")
    model.load_state_dict(torch.load(best_ckpt, map_location=device, weights_only=False)["model_state_dict"])
    model.eval()
    all_preds, all_probs, all_labels = [], [], []
    for imgs, labels in tqdm(test_loader, desc="  Testing", leave=False):
        imgs = imgs.to(device)
        with torch.no_grad(), autocast('cuda', enabled=True):
            logits = model(imgs)
        all_preds.append(logits.argmax(1).cpu().numpy())
        all_probs.append(F.softmax(logits, 1).cpu().numpy())
        all_labels.append(labels.numpy())
    preds = np.concatenate(all_preds); probs = np.concatenate(all_probs); labels = np.concatenate(all_labels)

    acc = accuracy_score(labels, preds)
    prec = precision_score(labels, preds, average=None, zero_division=0)
    rec = recall_score(labels, preds, average=None, zero_division=0)
    f1 = f1_score(labels, preds, average=None, zero_division=0)
    macro_f1 = f1_score(labels, preds, average="macro", zero_division=0)
    auc = roc_auc_score(labels, probs[:, 1]) if num_classes == 2 else roc_auc_score(labels, probs, multi_class="ovr", average="macro")
    cm = confusion_matrix(labels, preds)

    print(f"\n  Test Accuracy:   {acc*100:.2f}%")
    print(f"  Macro F1-Score:  {macro_f1*100:.2f}%")
    print(f"  AUC-ROC:         {auc:.4f}")
    print(f"  Training Time:   {total_time/60:.1f} min")
    print(f"\n{classification_report(labels, preds, target_names=train_ds.classes, digits=4)}")

    # Save CSVs
    rd = config["paths"]["results_dir"]
    # final_results.csv
    rows = [
        {"metric_name": "Accuracy", "overall_value": f"{acc:.3f}", "interpretation": "Overall correctness"},
        {"metric_name": "Macro F1-Score", "overall_value": f"{macro_f1:.3f}", "interpretation": "Balanced performance"},
        {"metric_name": "AUC-ROC", "overall_value": f"{auc:.3f}", "interpretation": "Threshold-independent metric"},
        {"metric_name": "Training Time", "overall_value": f"{total_time:.1f}s", "interpretation": "Total training duration"},
    ]
    for i, cls in enumerate(train_ds.classes):
        rows.append({"metric_name": f"Precision ({cls})", "overall_value": f"{prec[i]:.3f}", "interpretation": f"Precision for {cls}"})
        rows.append({"metric_name": f"Recall ({cls})", "overall_value": f"{rec[i]:.3f}", "interpretation": f"Recall for {cls}"})
        rows.append({"metric_name": f"F1-Score ({cls})", "overall_value": f"{f1[i]:.3f}", "interpretation": f"F1 for {cls}"})
    pd.DataFrame(rows).to_csv(os.path.join(rd, "final_results.csv"), index=False)

    # model_performance_analysis.csv
    perf_rows = []
    for i in range(len(history["train_loss"])):
        perf_rows.append({
            "epoch": i+1, "train_loss": f"{history['train_loss'][i]:.6f}",
            "val_loss": f"{history['val_loss'][i]:.6f}", "train_accuracy": f"{history['train_acc'][i]:.2f}",
            "val_accuracy": f"{history['val_acc'][i]:.2f}",
            "overfitting_gap": f"{history['val_loss'][i]-history['train_loss'][i]:.6f}",
            "learning_rate": f"{history['lr'][i]:.2e}",
        })
    pd.DataFrame(perf_rows).to_csv(os.path.join(rd, "model_performance_analysis.csv"), index=False)

    # Confusion matrix plot
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=train_ds.classes, yticklabels=train_ds.classes, ax=ax)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual"); ax.set_title("FractureMamba-ViT Confusion Matrix")
    plt.tight_layout(); plt.savefig(os.path.join(rd, "confusion_matrix.png"), dpi=150); plt.close()

    # Training curves
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    ep = range(1, len(history["train_loss"])+1)
    axes[0].plot(ep, history["train_loss"], "b-", label="Train"); axes[0].plot(ep, history["val_loss"], "r-", label="Val")
    axes[0].set_title("Loss"); axes[0].legend(); axes[0].grid(True, alpha=0.3)
    axes[1].plot(ep, history["train_acc"], "b-", label="Train"); axes[1].plot(ep, history["val_acc"], "r-", label="Val")
    axes[1].set_title("Accuracy (%)"); axes[1].legend(); axes[1].grid(True, alpha=0.3)
    plt.suptitle("FractureMamba-ViT Training Curves"); plt.tight_layout()
    plt.savefig(os.path.join(rd, "training_curves.png"), dpi=150); plt.close()

    print(f"\n{'='*60}")
    print(f"  DONE! Best Val Acc: {best_val_acc:.2f}% (epoch {best_epoch})")
    print(f"  Results saved to: {rd}")
    print(f"  Checkpoints saved to: {config['paths']['checkpoint_dir']}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
