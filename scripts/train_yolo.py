"""
train_yolo.py — Train YOLO for fracture detection on FracAtlas dataset.

Fresh training from COCO-pretrained YOLOv8l weights.
Matches fracture_yolo3 config but with reduced batch size to avoid OOM.

Dataset: data/fracatlas_yolo  (3266 train / 418 val / 399 test)
Classes: 1 — "fractured"
"""

import os
import sys
import shutil
from pathlib import Path

# ── Resolve project root (one level up from scripts/) ──────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
os.chdir(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT))

# ── Imports ─────────────────────────────────────────────────────────────────
from ultralytics import YOLO
import torch


def clean_label_cache(data_dir: Path):
    """Remove stale .cache files that can cause training bugs."""
    for cache_file in data_dir.rglob("*.cache"):
        print(f"  🗑  Removing stale cache: {cache_file}")
        cache_file.unlink()


def verify_dataset(data_dir: Path):
    """Quick sanity check on dataset structure and label format."""
    for split in ["train", "val", "test"]:
        img_dir = data_dir / "images" / split
        lbl_dir = data_dir / "labels" / split

        if not img_dir.exists():
            raise FileNotFoundError(f"Missing image directory: {img_dir}")

        n_images = len(list(img_dir.iterdir()))
        n_labels = len(list(lbl_dir.iterdir())) if lbl_dir.exists() else 0

        # Count positive (has bbox) vs negative (empty) labels
        n_positive = 0
        if lbl_dir.exists():
            for lbl in lbl_dir.iterdir():
                content = lbl.read_text().strip()
                if content:
                    n_positive += 1

        print(f"  {split:>5s}: {n_images:>4d} images | {n_labels:>4d} labels "
              f"({n_positive} positive, {n_labels - n_positive} negative)")

    print()


def main():
    print("=" * 60)
    print("  YOLO Fracture Detection — YOLOv8l Training")
    print("=" * 60)

    # ── Device selection ────────────────────────────────────────────────
    if torch.cuda.is_available():
        device = "0"
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"\n🖥  GPU: {gpu_name} ({vram_gb:.1f} GB VRAM)")
    else:
        device = "cpu"
        print("\n⚠  No GPU found — training on CPU (will be slow)")

    # ── Paths ───────────────────────────────────────────────────────────
    data_dir = PROJECT_ROOT / "data" / "fracatlas_yolo"
    config_path = PROJECT_ROOT / "configs" / "fracatlas_yolo.yaml"
    weights_path = PROJECT_ROOT / "weights" / "yolov8l.pt"

    if not weights_path.exists():
        print(f"\n⚠  Pretrained weights not found at {weights_path}")
        print("   Will download yolov8l.pt automatically from Ultralytics hub.")
        weights_path = "yolov8l.pt"  # Ultralytics will auto-download

    # ── Clean caches & verify dataset ───────────────────────────────────
    print("\n📁 Cleaning stale caches...")
    clean_label_cache(data_dir)

    print("\n📊 Dataset summary:")
    verify_dataset(data_dir)

    # ── Load COCO-pretrained YOLOv8 large ────────────────────────────────
    print("🔧 Loading fresh COCO-pretrained YOLOv8l...")
    model = YOLO(str(weights_path))

    # ── Training configuration ──────────────────────────────────────────
    # Matches fracture_yolo3 config (YOLOv8l + AdamW) but with:
    #   - batch=4 instead of 16 to prevent OOM/memory crash
    #   - workers=0 to prevent OpenCV memory crash on Windows

    run_name = "fracture_yolo4"

    train_args = dict(
        # ── Core ────────────────────────────────────────────────────────
        data=str(config_path),
        epochs=100,
        patience=20,              # same as yolo3
        imgsz=640,
        batch=4,                  # reduced from 16 to avoid OOM crash
        workers=0,                # 0 prevents OpenCV memory crash on Windows
        device=device,

        # ── Optimizer (matching yolo3: AdamW) ──────────────────────────
        optimizer="AdamW",
        lr0=0.001,                # same as yolo3
        lrf=0.01,                 # final LR = lr0 * lrf
        momentum=0.937,
        weight_decay=5e-4,
        warmup_epochs=3,          # same as yolo3
        warmup_momentum=0.8,
        warmup_bias_lr=0.1,
        cos_lr=True,              # cosine annealing schedule

        # ── Augmentation (matching yolo3) ───────────────────────────────
        hsv_h=0.01,               # same as yolo3
        hsv_s=0.3,                # same as yolo3
        hsv_v=0.4,                # same as yolo3
        degrees=15.0,             # same as yolo3
        translate=0.1,
        scale=0.5,                # same as yolo3
        shear=0.0,                # same as yolo3
        perspective=0.0,
        fliplr=0.5,
        flipud=0.1,               # same as yolo3
        mosaic=1.0,
        mixup=0.1,                # same as yolo3
        copy_paste=0.0,
        erasing=0.4,              # same as yolo3

        # ── Loss & detection ────────────────────────────────────────────
        box=7.5,
        cls=0.5,
        dfl=1.5,

        # ── Saving & output ─────────────────────────────────────────────
        project="runs/detect",
        name=run_name,
        exist_ok=True,            # allow resume
        save=True,
        save_period=-1,
        plots=True,
        val=True,
        verbose=True,

        # ── NMS ─────────────────────────────────────────────────────────
        conf=0.001,               # low conf for accurate mAP during training
        iou=0.6,                  # NMS IoU threshold

        # ── Regularization ──────────────────────────────────────────────
        dropout=0.0,
        nbs=64,
    )

    print(f"\n🚀 Starting training: {run_name}")
    print(f"   Epochs: {train_args['epochs']} | Batch: {train_args['batch']} | "
          f"ImgSz: {train_args['imgsz']} | LR: {train_args['lr0']}")
    print(f"   Output: runs/detect/{run_name}/")
    print("-" * 60)

    results = model.train(**train_args)

    # ── Post-training validation ────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  ✅  Training Complete!")
    print("=" * 60)

    best_weights = Path("runs/detect") / run_name / "weights" / "best.pt"
    print(f"\n📦 Best model saved to: {best_weights}")

    # Run validation on val set using best weights
    print("\n📈 Validation results (best.pt on val set):")
    best_model = YOLO(str(best_weights))
    metrics = best_model.val(
        data=str(config_path),
        imgsz=640,
        batch=4,
        device=device,
        conf=0.25,
        iou=0.5,
        plots=True,
        save_json=True,
    )

    print(f"\n  {'Metric':<15s} {'Value':>8s}")
    print(f"  {'-'*24}")
    print(f"  {'mAP@50':<15s} {metrics.box.map50:>8.4f}")
    print(f"  {'mAP@50-95':<15s} {metrics.box.map:>8.4f}")
    print(f"  {'Precision':<15s} {metrics.box.mp:>8.4f}")
    print(f"  {'Recall':<15s} {metrics.box.mr:>8.4f}")

    # ── Run test set evaluation ─────────────────────────────────────────
    print("\n📈 Test results (best.pt on test set):")
    test_metrics = best_model.val(
        data=str(config_path),
        split="test",
        imgsz=640,
        batch=4,
        device=device,
        conf=0.25,
        iou=0.5,
        plots=True,
    )

    print(f"\n  {'Metric':<15s} {'Value':>8s}")
    print(f"  {'-'*24}")
    print(f"  {'mAP@50':<15s} {test_metrics.box.map50:>8.4f}")
    print(f"  {'mAP@50-95':<15s} {test_metrics.box.map:>8.4f}")
    print(f"  {'Precision':<15s} {test_metrics.box.mp:>8.4f}")
    print(f"  {'Recall':<15s} {test_metrics.box.mr:>8.4f}")

    print("\n🎉 Done! All results saved under runs/detect/")


if __name__ == "__main__":
    main()
