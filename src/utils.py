import os
import random
import time
import yaml
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns


def load_config(path="config.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)


def seed_everything(seed=42, deterministic=True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic


def get_device():
    if torch.cuda.is_available():
        d = torch.device("cuda")
        p = torch.cuda.get_device_properties(0)
        print(f"[INFO] Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"[INFO] GPU Memory: {p.total_memory / 1024**3:.1f} GB")
    else:
        d = torch.device("cpu")
        print("[WARN] No GPU found — CPU training will be slow")
    return d


class AverageMeter:
    def __init__(self):
        self.val = self.avg = self.sum = self.count = 0

    def reset(self):
        self.val = self.avg = self.sum = self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class EarlyStopping:
    def __init__(self, patience=15, min_delta=0.001, mode="min"):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best = None

    def __call__(self, score):
        if self.best is None:
            self.best = score
            return False
        better = score < self.best - self.min_delta if self.mode == "min" else score > self.best + self.min_delta
        if better:
            self.best = score
            self.counter = 0
        else:
            self.counter += 1
        return self.counter >= self.patience


def save_checkpoint(model, optimizer, scheduler, epoch, val_acc, val_loss, path, is_best=False, scaler=None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    ckpt = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
        "val_acc": val_acc,
        "val_loss": val_loss,
    }
    if scaler:
        ckpt["scaler_state_dict"] = scaler.state_dict()
    torch.save(ckpt, path)
    if is_best:
        torch.save(ckpt, path.replace(".pth", "_best.pth"))
        print(f"  [SAVE] Best model saved (val_acc={val_acc:.4f})")


def load_checkpoint(model, path, optimizer=None, scheduler=None, scaler=None):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    if optimizer and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    if scheduler and ckpt.get("scheduler_state_dict"):
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    if scaler and "scaler_state_dict" in ckpt:
        scaler.load_state_dict(ckpt["scaler_state_dict"])
    ep, acc = ckpt.get("epoch", 0), ckpt.get("val_acc", 0.0)
    print(f"[INFO] Loaded checkpoint (epoch={ep}, val_acc={acc:.4f})")
    return ep, acc


def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[INFO] Total parameters: {total:,}")
    print(f"[INFO] Trainable parameters: {trainable:,}")
    return total, trainable


def get_model_size_mb(model):
    bytes_ = sum(p.nelement() * p.element_size() for p in model.parameters()) + \
             sum(b.nelement() * b.element_size() for b in model.buffers())
    return bytes_ / 1024 / 1024


def plot_training_curves(history, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("FractureMamba-ViT — Training Summary", fontsize=15, fontweight="bold")
    epochs = range(1, len(history["train_loss"]) + 1)

    for ax, y1k, y2k, title, ylabel in [
        (axes[0, 0], "train_loss", "val_loss", "Loss", "Loss"),
        (axes[0, 1], "train_acc", "val_acc", "Accuracy", "Accuracy (%)"),
    ]:
        ax.plot(epochs, history[y1k], "b-", label="Train", linewidth=2)
        ax.plot(epochs, history[y2k], "r-", label="Val",   linewidth=2)
        ax.set(xlabel="Epoch", ylabel=ylabel, title=title)
        ax.legend(); ax.grid(True, alpha=0.3)

    gap = [v - t for t, v in zip(history["train_loss"], history["val_loss"])]
    axes[1, 0].plot(epochs, gap, "g-", linewidth=2)
    axes[1, 0].axhline(0, color="k", linestyle="--", alpha=0.5)
    axes[1, 0].fill_between(epochs, gap, 0, alpha=0.2, color="green")
    axes[1, 0].set(xlabel="Epoch", ylabel="val_loss − train_loss", title="Overfitting Gap")
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].plot(epochs, history["lr"], "m-", linewidth=2)
    axes[1, 1].set(xlabel="Epoch", ylabel="LR", title="Learning Rate", yscale="log")
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Training curves saved to {save_path}")


def plot_confusion_matrix(cm, class_names, save_path):
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names,
                ax=ax, linewidths=0.5, linecolor="gray")
    ax.set(xlabel="Predicted", ylabel="Actual", title="Confusion Matrix — FractureMamba-ViT")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Confusion matrix saved to {save_path}")


class Timer:
    def __init__(self):
        self.start_time = None

    def start(self):
        self.start_time = time.time()

    def stop(self):
        return time.time() - self.start_time

    def __enter__(self):
        self.start(); return self

    def __exit__(self, *args):
        self.elapsed = self.stop()


def ensure_dirs(config):
    for key in ["checkpoint_dir", "results_dir", "explainability_dir", "logs_dir"]:
        os.makedirs(config["paths"][key], exist_ok=True)
