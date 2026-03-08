
import os, sys
sys.path.insert(0, os.path.dirname(__file__))
import time
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
from torch.optim.swa_utils import AveragedModel, SWALR
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from tqdm import tqdm

from utils import (
    load_config, seed_everything, get_device, AverageMeter,
    EarlyStopping, save_checkpoint, count_parameters,
    get_model_size_mb, plot_training_curves, ensure_dirs, Timer
)
from data_loader import (
    BoneFractureDataset, get_train_transforms, get_val_transforms, MixupCutmix, get_dataloaders
)
from model import FractureMambaViT, FocalLoss, build_model


def train_one_epoch(model, loader, criterion, optimizer, scaler, device, config, mixup_fn=None, epoch=0):
    model.train()
    loss_meter = AverageMeter("Loss")
    acc_meter = AverageMeter("Acc")
    
    accum_steps = config["training"]["gradient_accumulation_steps"]
    use_amp = config["training"]["mixed_precision"]
    clip_norm = config["training"]["gradient_clip_norm"]
    
    optimizer.zero_grad()
    
    pbar = tqdm(loader, desc=f"  Train Epoch {epoch+1}", leave=False, ncols=100)
    
    for step, (images, labels) in enumerate(pbar):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        
        # Apply MixUp/CutMix
        use_soft_labels = False
        if mixup_fn is not None and np.random.random() < config["augmentation"]["train"].get("mixup_cutmix_prob", 0.5):
            images, labels_mixed = mixup_fn(images, labels)
            use_soft_labels = True
        
        # Forward pass with mixed precision
        with autocast('cuda', enabled=use_amp):
            logits = model(images)
            if use_soft_labels:
                loss = criterion(logits, labels_mixed)
            else:
                loss = criterion(logits, labels)
            loss = loss / accum_steps  # Scale for accumulation
        
        # Backward pass
        scaler.scale(loss).backward()
        
        # Gradient accumulation
        if (step + 1) % accum_steps == 0 or (step + 1) == len(loader):
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
        
        # Metrics (use original labels for accuracy)
        with torch.no_grad():
            preds = logits.argmax(dim=1)
            if use_soft_labels:
                # For mixed labels, use argmax of soft labels
                true_labels = labels_mixed.argmax(dim=1)
            else:
                true_labels = labels
            correct = (preds == true_labels).float().mean().item() * 100
        
        loss_meter.update(loss.item() * accum_steps, images.size(0))
        acc_meter.update(correct, images.size(0))
        
        pbar.set_postfix({"loss": f"{loss_meter.avg:.4f}", "acc": f"{acc_meter.avg:.1f}%"})
    
    return loss_meter.avg, acc_meter.avg


@torch.no_grad()
def validate(model, loader, criterion, device, config):
    model.eval()
    loss_meter = AverageMeter("Loss")
    acc_meter = AverageMeter("Acc")
    
    use_amp = config["training"]["mixed_precision"]
    
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        
        with autocast('cuda', enabled=use_amp):
            logits = model(images)
            loss = criterion(logits, labels)
        
        preds = logits.argmax(dim=1)
        correct = (preds == labels).float().mean().item() * 100
        
        loss_meter.update(loss.item(), images.size(0))
        acc_meter.update(correct, images.size(0))
    
    return loss_meter.avg, acc_meter.avg


def train(config, fold=None):
    seed_everything(config["seed"], config["deterministic"])
    device = get_device()
    ensure_dirs(config)
    
    train_cfg = config["training"]
    num_epochs = train_cfg["epochs"]
    
    # Data
    print("\n" + "="*70)
    print("  LOADING DATA")
    print("="*70)
    train_loader, val_loader, test_loader, class_names, class_weights = get_dataloaders(config)
    num_classes = len(class_names)
    config["data"]["num_classes"] = num_classes
    
    # Model
    print("\n" + "="*70)
    print("  BUILDING MODEL: FractureMamba-ViT")
    print("="*70)
    model = build_model(config)
    model = model.to(device)
    total_params, trainable_params = count_parameters(model)
    model_size = get_model_size_mb(model)
    print(f"[INFO] Model size: {model_size:.1f} MB")
    
    # Loss function
    focal_cfg = train_cfg["focal_loss"]
    alpha = class_weights if focal_cfg.get("alpha") is None else torch.tensor(focal_cfg["alpha"])
    criterion = FocalLoss(
        alpha=alpha,
        gamma=focal_cfg["gamma"],
        label_smoothing=train_cfg["label_smoothing"],
    )
    
    # Optimizer
    opt_cfg = train_cfg["optimizer"]
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=opt_cfg["lr"],
        weight_decay=opt_cfg["weight_decay"],
        betas=tuple(opt_cfg["betas"]),
    )
    
    # Scheduler
    sched_cfg = train_cfg["scheduler"]
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=sched_cfg["T_0"],
        T_mult=sched_cfg["T_mult"],
        eta_min=sched_cfg["eta_min"],
    )
    
    # Mixed precision
    scaler = GradScaler(enabled=train_cfg["mixed_precision"])
    
    # MixUp/CutMix
    aug_cfg = config["augmentation"]["train"]
    mixup_fn = None
    if aug_cfg.get("mixup", {}).get("enabled", False) or aug_cfg.get("cutmix", {}).get("enabled", False):
        mixup_fn = MixupCutmix(
            mixup_alpha=aug_cfg.get("mixup", {}).get("alpha", 0.4),
            cutmix_alpha=aug_cfg.get("cutmix", {}).get("alpha", 1.0),
            prob=aug_cfg.get("mixup_cutmix_prob", 0.5),
            num_classes=num_classes,
        )
    
    # SWA
    swa_cfg = train_cfg.get("swa", {})
    use_swa = swa_cfg.get("enabled", False)
    swa_model = None
    swa_scheduler = None
    if use_swa:
        swa_model = AveragedModel(model)
        swa_start = swa_cfg.get("start_epoch", 80)
        swa_lr = swa_cfg.get("lr", 5e-5)
        swa_scheduler = SWALR(optimizer, swa_lr=swa_lr, anneal_epochs=5)
    
    # Early stopping
    es_cfg = train_cfg.get("early_stopping", {})
    early_stopper = EarlyStopping(
        patience=es_cfg.get("patience", 20),
        min_delta=es_cfg.get("min_delta", 0.001),
        mode="max",
    )
    
    # Warmup
    warmup_cfg = train_cfg.get("warmup", {})
    warmup_epochs = warmup_cfg.get("epochs", 5)
    warmup_start_lr = warmup_cfg.get("start_lr", 1e-6)
    target_lr = opt_cfg["lr"]
    
    # Training history
    history = {
        "train_loss": [], "val_loss": [],
        "train_acc": [], "val_acc": [],
        "lr": [],
    }
    
    best_val_acc = 0.0
    best_epoch = 0
    fold_str = f" (Fold {fold})" if fold is not None else ""
    
    checkpoint_name = f"checkpoint_fold{fold}.pth" if fold is not None else "checkpoint.pth"
    checkpoint_path = os.path.join(config["paths"]["checkpoint_dir"], checkpoint_name)
    
    print(f"\n{'='*70}")
    print(f"  TRAINING{fold_str} — {num_epochs} epochs")
    print(f"  Batch size: {train_cfg['batch_size']} × {train_cfg['gradient_accumulation_steps']} = {train_cfg['batch_size'] * train_cfg['gradient_accumulation_steps']} effective")
    print(f"  Mixed precision: {train_cfg['mixed_precision']}")
    print(f"  SWA: {use_swa} (starts epoch {swa_cfg.get('start_epoch', 'N/A')})")
    print(f"{'='*70}\n")
    
    training_timer = Timer()
    training_timer.start()
    
    for epoch in range(num_epochs):
        epoch_start = time.time()
        
        # Warmup LR
        if epoch < warmup_epochs:
            warmup_lr = warmup_start_lr + (target_lr - warmup_start_lr) * (epoch / warmup_epochs)
            for pg in optimizer.param_groups:
                pg["lr"] = warmup_lr
        
        current_lr = optimizer.param_groups[0]["lr"]
        
        # Train
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, device, config, mixup_fn, epoch
        )
        
        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, device, config)
        
        # Update scheduler
        if epoch >= warmup_epochs:
            if use_swa and epoch >= swa_start:
                swa_model.update_parameters(model)
                swa_scheduler.step()
            else:
                scheduler.step()
        
        # Record history
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        history["lr"].append(current_lr)
        
        epoch_time = time.time() - epoch_start
        overfitting_gap = val_loss - train_loss
        
        # Print epoch results
        print(f"Epoch {epoch+1:3d}/{num_epochs} | "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.1f}% | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.1f}% | "
              f"Gap: {overfitting_gap:.4f} | LR: {current_lr:.2e} | "
              f"Time: {epoch_time:.1f}s")
        
        # Save best model
        is_best = val_acc > best_val_acc
        if is_best:
            best_val_acc = val_acc
            best_epoch = epoch + 1
        
        save_checkpoint(
            model, optimizer, scheduler, epoch, val_acc, val_loss,
            checkpoint_path, is_best=is_best, scaler=scaler
        )
        
        # Early stopping
        if early_stopper(val_acc):
            print(f"\n  [EARLY STOP] No improvement for {es_cfg.get('patience', 20)} epochs. Stopping.")
            break
    
    total_training_time = training_timer.stop()
    
    # If using SWA, update batch norm
    if use_swa and swa_model is not None:
        print("\n[SWA] Updating batch normalization statistics...")
        torch.optim.swa_utils.update_bn(train_loader, swa_model, device=device)
        # Save SWA model
        swa_path = checkpoint_path.replace(".pth", "_swa.pth")
        torch.save({"model_state_dict": swa_model.module.state_dict()}, swa_path)
        print(f"  [SAVE] SWA model saved to {swa_path}")
    
    # Generate model_performance_analysis.csv
    _save_performance_analysis(history, best_val_acc, best_epoch, total_training_time, config, fold)
    
    # Plot training curves
    curves_path = os.path.join(config["paths"]["results_dir"], 
                                f"training_curves{'_fold'+str(fold) if fold else ''}.png")
    plot_training_curves(history, curves_path)
    
    print(f"\n{'='*70}")
    print(f"  TRAINING COMPLETE{fold_str}")
    print(f"  Best Val Accuracy: {best_val_acc:.2f}% (Epoch {best_epoch})")
    print(f"  Total Training Time: {total_training_time/60:.1f} minutes")
    print(f"{'='*70}\n")
    
    return best_val_acc, best_epoch, history, total_training_time


def _save_performance_analysis(history, best_val_acc, best_epoch, total_time, config, fold=None):
    results_dir = config["paths"]["results_dir"]
    os.makedirs(results_dir, exist_ok=True)
    
    suffix = f"_fold{fold}" if fold is not None else ""
    
    rows = []
    for i in range(len(history["train_loss"])):
        rows.append({
            "epoch": i + 1,
            "train_loss": f"{history['train_loss'][i]:.6f}",
            "val_loss": f"{history['val_loss'][i]:.6f}",
            "train_accuracy": f"{history['train_acc'][i]:.2f}",
            "val_accuracy": f"{history['val_acc'][i]:.2f}",
            "overfitting_gap": f"{history['val_loss'][i] - history['train_loss'][i]:.6f}",
            "learning_rate": f"{history['lr'][i]:.2e}",
        })
    
    df = pd.DataFrame(rows)
    csv_path = os.path.join(results_dir, f"model_performance_analysis{suffix}.csv")
    df.to_csv(csv_path, index=False)
    
    # Append summary
    max_gap = max(history["val_loss"][i] - history["train_loss"][i] for i in range(len(history["train_loss"])))
    
    with open(csv_path, "a") as f:
        f.write(f"\n# GENERALIZATION METRICS\n")
        f.write(f"# Max Overfitting Gap: {max_gap:.4f}\n")
        f.write(f"# Best Val Accuracy: {best_val_acc:.2f}% (epoch {best_epoch})\n")
        f.write(f"# Total Training Time: {total_time/60:.1f} minutes\n")
    
    print(f"[INFO] Performance analysis saved to {csv_path}")


def cross_validate(config):
    seed_everything(config["seed"])
    
    print("\n" + "="*70)
    print("  5-FOLD STRATIFIED CROSS-VALIDATION")
    print("="*70)
    
    # Load full dataset
    root_dir = config["data"]["root_dir"]
    train_transform = get_train_transforms(config)
    val_transform = get_val_transforms(config)
    
    full_dataset = BoneFractureDataset(root_dir, split="train", transform=train_transform, config=config)
    all_labels = full_dataset.labels
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=config["seed"])
    
    fold_accuracies = []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(range(len(full_dataset)), all_labels)):
        print(f"\n{'='*50}")
        print(f"  FOLD {fold + 1} / 5")
        print(f"  Train: {len(train_idx)} | Val: {len(val_idx)}")
        print(f"{'='*50}")
        
        # Create subsets
        train_subset = Subset(full_dataset, train_idx)
        
        # Val subset uses val transforms
        val_dataset_fold = BoneFractureDataset(root_dir, split="train", transform=val_transform, config=config)
        val_subset = Subset(val_dataset_fold, val_idx)
        
        # Samplers
        train_labels_fold = all_labels[train_idx]
        class_counts = np.bincount(train_labels_fold, minlength=len(full_dataset.classes))
        class_w = 1.0 / (class_counts + 1e-6)
        class_w = class_w / class_w.sum()
        sample_w = class_w[train_labels_fold]
        sampler = WeightedRandomSampler(torch.DoubleTensor(sample_w), len(sample_w), replacement=True)
        
        batch_size = config["training"]["batch_size"]
        num_workers = config["data"].get("num_workers", 4)
        
        train_loader = DataLoader(train_subset, batch_size=batch_size, sampler=sampler,
                                   num_workers=num_workers, pin_memory=True, drop_last=True)
        val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False,
                                num_workers=num_workers, pin_memory=True)
        
        # Build fresh model for each fold
        device = get_device()
        model = build_model(config).to(device)
        
        # Loss
        focal_cfg = config["training"]["focal_loss"]
        alpha = torch.FloatTensor(len(full_dataset.classes))
        for i in range(len(full_dataset.classes)):
            alpha[i] = 1.0 / (class_counts[i] + 1e-6)
        alpha = alpha / alpha.sum() * len(full_dataset.classes)
        
        criterion = FocalLoss(
            alpha=alpha, gamma=focal_cfg["gamma"],
            label_smoothing=config["training"]["label_smoothing"],
        )
        
        # Optimizer & scheduler
        opt_cfg = config["training"]["optimizer"]
        optimizer = torch.optim.AdamW(model.parameters(), lr=opt_cfg["lr"],
                                       weight_decay=opt_cfg["weight_decay"],
                                       betas=tuple(opt_cfg["betas"]))
        
        sched_cfg = config["training"]["scheduler"]
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=sched_cfg["T_0"], T_mult=sched_cfg["T_mult"], eta_min=sched_cfg["eta_min"]
        )
        
        scaler = GradScaler(enabled=config["training"]["mixed_precision"])
        
        # Reduced epochs for CV
        cv_epochs = min(50, config["training"]["epochs"])
        best_val_acc = 0.0
        
        aug_cfg = config["augmentation"]["train"]
        mixup_fn = MixupCutmix(
            mixup_alpha=aug_cfg.get("mixup", {}).get("alpha", 0.4),
            cutmix_alpha=aug_cfg.get("cutmix", {}).get("alpha", 1.0),
            prob=aug_cfg.get("mixup_cutmix_prob", 0.5),
            num_classes=len(full_dataset.classes),
        )
        
        warmup_epochs = config["training"].get("warmup", {}).get("epochs", 5)
        warmup_start_lr = config["training"].get("warmup", {}).get("start_lr", 1e-6)
        target_lr = opt_cfg["lr"]
        
        for epoch in range(cv_epochs):
            if epoch < warmup_epochs:
                warmup_lr = warmup_start_lr + (target_lr - warmup_start_lr) * (epoch / warmup_epochs)
                for pg in optimizer.param_groups:
                    pg["lr"] = warmup_lr
            
            train_loss, train_acc = train_one_epoch(
                model, train_loader, criterion, optimizer, scaler, device, config, mixup_fn, epoch
            )
            val_loss, val_acc = validate(model, val_loader, criterion, device, config)
            
            if epoch >= warmup_epochs:
                scheduler.step()
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
            
            if (epoch + 1) % 10 == 0:
                print(f"  Fold {fold+1} Epoch {epoch+1}/{cv_epochs} | "
                      f"Train Acc: {train_acc:.1f}% | Val Acc: {val_acc:.1f}% | Best: {best_val_acc:.1f}%")
        
        fold_accuracies.append(best_val_acc)
        print(f"\n  Fold {fold+1} Best Val Accuracy: {best_val_acc:.2f}%")
    
    mean_acc = np.mean(fold_accuracies)
    std_acc = np.std(fold_accuracies)
    
    print(f"\n{'='*70}")
    print(f"  CROSS-VALIDATION RESULTS")
    print(f"  Fold Accuracies: {[f'{a:.2f}%' for a in fold_accuracies]}")
    print(f"  Mean ± Std: {mean_acc:.2f}% ± {std_acc:.2f}%")
    print(f"{'='*70}\n")
    
    return mean_acc, std_acc, fold_accuracies


def main():
    parser = argparse.ArgumentParser(description="Train FractureMamba-ViT")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file")
    parser.add_argument("--epochs", type=int, default=None, help="Override number of epochs")
    parser.add_argument("--cv", action="store_true", help="Run 5-fold cross-validation")
    parser.add_argument("--debug", action="store_true", help="Debug mode (2 epochs, small subset)")
    args = parser.parse_args()
    
    config = load_config(args.config)
    
    if args.epochs:
        config["training"]["epochs"] = args.epochs
    
    if args.debug:
        config["training"]["epochs"] = 2
        config["training"]["batch_size"] = 4
        print("[DEBUG] Running in debug mode (2 epochs, batch_size=4)")
    
    if args.cv:
        mean_acc, std_acc, fold_accs = cross_validate(config)
    else:
        best_acc, best_epoch, history, training_time = train(config)


if __name__ == "__main__":
    main()
