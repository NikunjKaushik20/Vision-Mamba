import os
import sys
import torch
import numpy as np
import cv2
from tqdm import tqdm
from pathlib import Path
from sklearn.metrics import accuracy_score, confusion_matrix

sys.path.insert(0, os.path.dirname(__file__))

from utils import load_config, get_device, get_model_size_mb, plot_confusion_matrix, load_checkpoint
from data_loader import get_val_transforms
from model import build_model

def main():
    config_path = "d:/Vision-Mamba/config.yaml"
    config = load_config(config_path)
    device = get_device()
    
    # Force test configuration
    real_life_dir = "d:/Vision-Mamba/real_life_test_cropped"
    
    print(f"[INFO] Initializing Real-Life Evaluation on {device}")
    
    # Model
    model = build_model(config).to(device)
    
    # Load checkpoint
    ckpt_dir = config["paths"]["checkpoint_dir"]
    robust_path = os.path.join(ckpt_dir, "checkpoint_robust_best.pth")
    best_path = os.path.join(ckpt_dir, "checkpoint_best.pth")
    swa_path = os.path.join(ckpt_dir, "checkpoint_swa.pth")
    
    ckpt_path = None
    if os.path.exists(robust_path):
        ckpt_path = robust_path
    elif os.path.exists(swa_path):
        ckpt_path = swa_path
    elif os.path.exists(best_path):
        ckpt_path = best_path
    else:
        print("[ERROR] No checkpoint found. Exiting.")
        return

    print(f"Loading checkpoint from {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=device)
    state_dict = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint
    
    # Handle DDP / DataParallel prefixes if present
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v
            
    # Load with strict=False to bypass slight mismatches or non-critical keys
    model.load_state_dict(new_state_dict, strict=False)

    model.eval()
    
    transform = get_val_transforms(config)
    class_names = config["data"]["class_names"] # ["fractured", "not fractured"]
    # Based on user input, all images are "fractured" (index 0)
    FRACTURED_CLASS_IDX = 0
    
    image_paths = [os.path.join(real_life_dir, f) for f in os.listdir(real_life_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    # Use TTA (Test-Time Augmentation)
    from data_loader import get_tta_transforms
    tta_transforms = get_tta_transforms(config)
    num_tta = len(tta_transforms)
    print(f"[INFO] Using Test-Time Augmentation with {num_tta} passes")
    
    all_preds = []
    all_labels = []
    
    print(f"[INFO] Found {len(image_paths)} images. Running evaluation...")
    
    with torch.no_grad():
        for img_path in tqdm(image_paths, desc="Evaluating"):
            # Load image
            image = cv2.imread(img_path)
            if image is None:
                continue
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Predict with TTA
            sum_probs = None
            for tta_idx, t_transform in enumerate(tta_transforms):
                transformed = t_transform(image=image)
                image_tensor = transformed["image"].unsqueeze(0).to(device)
                
                with torch.amp.autocast('cuda', enabled=config["training"]["mixed_precision"]):
                    logits = model(image_tensor)
                    probs = torch.nn.functional.softmax(logits, dim=1)
                
                if sum_probs is None:
                    sum_probs = probs
                else:
                    sum_probs += probs
                    
            avg_probs = sum_probs / num_tta
            pred = avg_probs.argmax(dim=1).item()
            
            all_preds.append(pred)
            all_labels.append(FRACTURED_CLASS_IDX) 
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    # Compute metrics
    acc = accuracy_score(all_labels, all_preds)
    cm = confusion_matrix(all_labels, all_preds, labels=[0, 1])
    
    res_text = f"\n{'='*50}\n"
    res_text += f"  REAL-LIFE TEST RESULTS\n"
    res_text += f"{'='*50}\n"
    res_text += f"  Total Images:      {len(all_labels)}\n"
    res_text += f"  Accuracy:          {acc*100:.2f}%\n"
    res_text += f"{'='*50}\n\n"
    res_text += f"Confusion Matrix:\n"
    res_text += f"True Fractured, Predicted Fractured (True Positives):       {cm[0, 0]}\n"
    res_text += f"True Fractured, Predicted Not Fractured (False Negatives):  {cm[0, 1]}\n"
    
    print(res_text)
    
    results_dir = config["paths"]["results_dir"]
    os.makedirs(results_dir, exist_ok=True)
    
    # Save text results
    with open(os.path.join(results_dir, "real_life_results.txt"), "w", encoding="utf-8") as f:
        f.write(res_text)
        
    cm_path = os.path.join(results_dir, "real_life_confusion_matrix.png")
    plot_confusion_matrix(cm, class_names, cm_path)
    
    print(f"\n[INFO] Confusion matrix plot saved to {cm_path}")
    print(f"[INFO] Text results saved to {os.path.join(results_dir, 'real_life_results.txt')}")

if __name__ == "__main__":
    main()
