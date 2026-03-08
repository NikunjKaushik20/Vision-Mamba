
import os, sys
sys.path.insert(0, os.path.dirname(__file__))
import time
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.amp import autocast
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score,
    roc_curve, auc,
)
from tqdm import tqdm

from utils import (
    load_config, seed_everything, get_device, get_model_size_mb,
    plot_confusion_matrix, ensure_dirs, Timer, load_checkpoint
)
from data_loader import get_dataloaders, get_tta_transforms, BoneFractureDataset
from model import build_model


@torch.no_grad()
def evaluate_model(model, loader, device, use_amp=True):
    model.eval()
    all_preds = []
    all_probs = []
    all_labels = []
    
    for images, labels in tqdm(loader, desc="  Evaluating", leave=False, ncols=100):
        images = images.to(device, non_blocking=True)
        
        with autocast('cuda', enabled=use_amp):
            logits = model(images)
        
        probs = F.softmax(logits, dim=1)
        preds = logits.argmax(dim=1)
        
        all_preds.append(preds.cpu().numpy())
        all_probs.append(probs.cpu().numpy())
        all_labels.append(labels.numpy())
    
    all_preds = np.concatenate(all_preds)
    all_probs = np.concatenate(all_probs)
    all_labels = np.concatenate(all_labels)
    
    return all_preds, all_probs, all_labels


@torch.no_grad()
def evaluate_with_tta(model, test_dataset, tta_transforms, device, batch_size=8, use_amp=True):
    model.eval()
    
    num_samples = len(test_dataset)
    num_tta = len(tta_transforms)
    all_probs_sum = None
    all_labels = None
    
    print(f"  [TTA] Running {num_tta} augmentation passes...")
    
    for tta_idx, transform in enumerate(tta_transforms):
        # Create dataset with this transform
        import cv2
        tta_dataset = BoneFractureDataset(
            root_dir=os.path.dirname(test_dataset.root_dir),
            split=test_dataset.split,
            transform=transform,
            config=test_dataset.config,
        )
        
        loader = torch.utils.data.DataLoader(
            tta_dataset, batch_size=batch_size, shuffle=False,
            num_workers=2, pin_memory=True,
        )
        
        preds, probs, labels = evaluate_model(model, loader, device, use_amp)
        
        if all_probs_sum is None:
            all_probs_sum = probs
            all_labels = labels
        else:
            all_probs_sum += probs
        
        acc = accuracy_score(labels, probs.argmax(axis=1)) * 100
        print(f"    TTA pass {tta_idx+1}/{num_tta}: {acc:.1f}% accuracy")
    
    # Average probabilities
    all_probs_avg = all_probs_sum / num_tta
    all_preds = all_probs_avg.argmax(axis=1)
    
    return all_preds, all_probs_avg, all_labels


def compute_all_metrics(preds, probs, labels, class_names):
    """Compute all required hackathon metrics."""
    num_classes = len(class_names)
    
    # Overall accuracy
    overall_acc = accuracy_score(labels, preds)
    
    # Per-class metrics
    precision = precision_score(labels, preds, average=None, zero_division=0)
    recall = recall_score(labels, preds, average=None, zero_division=0)
    f1 = f1_score(labels, preds, average=None, zero_division=0)
    
    # Macro averages
    macro_precision = precision_score(labels, preds, average="macro", zero_division=0)
    macro_recall = recall_score(labels, preds, average="macro", zero_division=0)
    macro_f1 = f1_score(labels, preds, average="macro", zero_division=0)
    weighted_f1 = f1_score(labels, preds, average="weighted", zero_division=0)
    
    # Confusion matrix
    cm = confusion_matrix(labels, preds)
    
    # AUC-ROC
    if num_classes == 2:
        auc_roc = roc_auc_score(labels, probs[:, 1])
    else:
        try:
            auc_roc = roc_auc_score(labels, probs, multi_class="ovr", average="macro")
        except ValueError:
            auc_roc = 0.0
    
    metrics = {
        "overall_accuracy": overall_acc,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "auc_roc": auc_roc,
        "per_class_precision": precision,
        "per_class_recall": recall,
        "per_class_f1": f1,
        "confusion_matrix": cm,
    }
    
    return metrics


def generate_final_results_csv(metrics, class_names, training_time, inference_time, model_size,
                                 cv_mean=None, cv_std=None, save_path="final_results.csv"):
    """Generate final_results.csv in the required format."""
    num_classes = len(class_names)
    
    rows = []
    
    # Accuracy
    row = {"metric_name": "Accuracy", "overall_value": f"{metrics['overall_accuracy']:.3f}"}
    for i, cls in enumerate(class_names):
        row[f"class_{i+1}_value ({cls})"] = "N/A"
    row["interpretation"] = "Overall correctness"
    rows.append(row)
    
    # Precision
    row = {"metric_name": "Precision", "overall_value": f"{metrics['macro_precision']:.3f}"}
    for i, cls in enumerate(class_names):
        row[f"class_{i+1}_value ({cls})"] = f"{metrics['per_class_precision'][i]:.3f}"
    row["interpretation"] = "Positive prediction reliability"
    rows.append(row)
    
    # Recall
    row = {"metric_name": "Recall", "overall_value": f"{metrics['macro_recall']:.3f}"}
    for i, cls in enumerate(class_names):
        row[f"class_{i+1}_value ({cls})"] = f"{metrics['per_class_recall'][i]:.3f}"
    row["interpretation"] = "Detection rate per fracture type"
    rows.append(row)
    
    # F1-Score
    row = {"metric_name": "F1-Score", "overall_value": f"{metrics['macro_f1']:.3f}"}
    for i, cls in enumerate(class_names):
        row[f"class_{i+1}_value ({cls})"] = f"{metrics['per_class_f1'][i]:.3f}"
    row["interpretation"] = "Balanced performance per class"
    rows.append(row)
    
    # Weighted F1
    row = {"metric_name": "Weighted F1-Score", "overall_value": f"{metrics['weighted_f1']:.3f}"}
    for i, cls in enumerate(class_names):
        row[f"class_{i+1}_value ({cls})"] = "N/A"
    row["interpretation"] = "Weighted average F1 across classes"
    rows.append(row)
    
    # AUC-ROC
    row = {"metric_name": "AUC-ROC", "overall_value": f"{metrics['auc_roc']:.3f}"}
    for i, cls in enumerate(class_names):
        row[f"class_{i+1}_value ({cls})"] = "N/A"
    row["interpretation"] = "Threshold-independent discriminative ability"
    rows.append(row)
    
    # Training time
    row = {"metric_name": "Training Time", "overall_value": f"{training_time:.1f}s"}
    for i, cls in enumerate(class_names):
        row[f"class_{i+1}_value ({cls})"] = "N/A"
    row["interpretation"] = "Total model training duration"
    rows.append(row)
    
    # Inference time
    row = {"metric_name": "Inference Time", "overall_value": f"{inference_time:.4f}s"}
    for i, cls in enumerate(class_names):
        row[f"class_{i+1}_value ({cls})"] = "N/A"
    row["interpretation"] = "Per-image inference latency"
    rows.append(row)
    
    # Model size
    row = {"metric_name": "Model Size", "overall_value": f"{model_size:.1f}MB"}
    for i, cls in enumerate(class_names):
        row[f"class_{i+1}_value ({cls})"] = "N/A"
    row["interpretation"] = "Model memory footprint"
    rows.append(row)
    
    # Cross-validation
    if cv_mean is not None and cv_std is not None:
        row = {"metric_name": "Cross-Validation", "overall_value": f"{cv_mean:.2f}% ± {cv_std:.2f}%"}
        for i, cls in enumerate(class_names):
            row[f"class_{i+1}_value ({cls})"] = "N/A"
        row["interpretation"] = "5-fold stratified CV mean ± std accuracy"
        rows.append(row)
    
    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
    df.to_csv(save_path, index=False)
    
    # Append confusion matrix
    with open(save_path, "a") as f:
        f.write(f"\n# Confusion Matrix\n")
        cm = metrics["confusion_matrix"]
        header = "," + ",".join(class_names)
        f.write(f"# {header}\n")
        for i, cls in enumerate(class_names):
            row_str = f"# {cls}," + ",".join(str(v) for v in cm[i])
            f.write(f"{row_str}\n")
    
    print(f"[INFO] Final results saved to {save_path}")


def measure_inference_time(model, device, img_size=224, num_runs=100):
    """Measure average inference time per image."""
    model.eval()
    dummy = torch.randn(1, 3, img_size, img_size).to(device)
    
    # Warmup
    for _ in range(10):
        with torch.no_grad():
            _ = model(dummy)
    
    # Measure
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    times = []
    for _ in range(num_runs):
        start = time.time()
        with torch.no_grad():
            _ = model(dummy)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        times.append(time.time() - start)
    
    avg_time = np.mean(times)
    print(f"[INFO] Average inference time: {avg_time*1000:.2f} ms/image")
    return avg_time


def main():
    parser = argparse.ArgumentParser(description="Evaluate FractureMamba-ViT")
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint")
    parser.add_argument("--tta", action="store_true", help="Use Test-Time Augmentation")
    parser.add_argument("--cv-mean", type=float, default=None, help="CV mean accuracy")
    parser.add_argument("--cv-std", type=float, default=None, help="CV std accuracy")
    parser.add_argument("--training-time", type=float, default=0.0, help="Total training time (seconds)")
    args = parser.parse_args()
    
    config = load_config(args.config)
    seed_everything(config["seed"])
    device = get_device()
    ensure_dirs(config)
    
    # Data
    train_loader, val_loader, test_loader, class_names, class_weights = get_dataloaders(config)
    num_classes = len(class_names)
    config["data"]["num_classes"] = num_classes
    
    # Model
    model = build_model(config).to(device)
    model_size = get_model_size_mb(model)
    
    # Load checkpoint
    if args.checkpoint:
        load_checkpoint(model, args.checkpoint)
    else:
        # Try to find best checkpoint
        ckpt_dir = config["paths"]["checkpoint_dir"]
        best_path = os.path.join(ckpt_dir, "checkpoint_best.pth")
        swa_path = os.path.join(ckpt_dir, "checkpoint_swa.pth")
        
        if os.path.exists(swa_path):
            load_checkpoint(model, swa_path)
        elif os.path.exists(best_path):
            load_checkpoint(model, best_path)
        else:
            print("[WARN] No checkpoint found. Using untrained model.")
    
    # Evaluate
    print("\n" + "="*70)
    print("  EVALUATING ON TEST SET")
    print("="*70)
    
    if args.tta or config["evaluation"]["tta"]["enabled"]:
        print("[INFO] Using Test-Time Augmentation (TTA)")
        tta_transforms = get_tta_transforms(config)
        test_dataset = test_loader.dataset
        preds, probs, labels = evaluate_with_tta(
            model, test_dataset, tta_transforms, device,
            batch_size=config["training"]["batch_size"],
            use_amp=config["training"]["mixed_precision"],
        )
    else:
        preds, probs, labels = evaluate_model(
            model, test_loader, device,
            use_amp=config["training"]["mixed_precision"],
        )
    
    # Compute metrics
    metrics = compute_all_metrics(preds, probs, labels, class_names)
    
    # Inference time
    inference_time = measure_inference_time(model, device, config["data"]["image_size"])
    
    # Print results
    print(f"\n{'='*70}")
    print(f"  TEST RESULTS — FractureMamba-ViT")
    print(f"{'='*70}")
    print(f"  Overall Accuracy:  {metrics['overall_accuracy']*100:.2f}%")
    print(f"  Macro Precision:   {metrics['macro_precision']*100:.2f}%")
    print(f"  Macro Recall:      {metrics['macro_recall']*100:.2f}%")
    print(f"  Macro F1-Score:    {metrics['macro_f1']*100:.2f}%")
    print(f"  Weighted F1-Score: {metrics['weighted_f1']*100:.2f}%")
    print(f"  AUC-ROC:           {metrics['auc_roc']:.4f}")
    print(f"  Inference Time:    {inference_time*1000:.2f} ms/image")
    print(f"  Model Size:        {model_size:.1f} MB")
    print(f"{'='*70}")
    
    print("\nPer-class metrics:")
    for i, cls in enumerate(class_names):
        print(f"  {cls}: P={metrics['per_class_precision'][i]:.3f}, "
              f"R={metrics['per_class_recall'][i]:.3f}, "
              f"F1={metrics['per_class_f1'][i]:.3f}")
    
    print(f"\nClassification Report:\n")
    print(classification_report(labels, preds, target_names=class_names, digits=4))
    
    # Save results
    results_dir = config["paths"]["results_dir"]
    
    # Confusion matrix heatmap
    cm_path = os.path.join(results_dir, "confusion_matrix.png")
    plot_confusion_matrix(metrics["confusion_matrix"], class_names, cm_path)
    
    # Final results CSV
    csv_path = os.path.join(results_dir, "final_results.csv")
    generate_final_results_csv(
        metrics, class_names,
        training_time=args.training_time,
        inference_time=inference_time,
        model_size=model_size,
        cv_mean=args.cv_mean,
        cv_std=args.cv_std,
        save_path=csv_path,
    )
    
    # Also save to project root
    root_csv = os.path.join(os.path.dirname(config["paths"]["results_dir"]), "final_results.csv")
    generate_final_results_csv(
        metrics, class_names,
        training_time=args.training_time,
        inference_time=inference_time,
        model_size=model_size,
        cv_mean=args.cv_mean,
        cv_std=args.cv_std,
        save_path=root_csv,
    )


if __name__ == "__main__":
    main()
