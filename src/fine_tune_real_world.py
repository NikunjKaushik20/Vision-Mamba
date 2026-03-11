import os
import sys
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.amp import autocast, GradScaler
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(__file__))
from utils import load_config, seed_everything, get_device, save_checkpoint
from data_loader import get_dataloaders
from model import build_model, FocalLoss

def main():
    config = load_config("d:/Vision-Mamba/config.yaml")
    seed_everything(config["seed"])
    device = get_device()
    
    # Force heavy augmentations for real-world robustness training
    print("[INFO] Loading datasets with Real-World Domain Adaptation augmentations active...")
    train_loader, val_loader, _, _, class_weights = get_dataloaders(config)
    
    model = build_model(config).to(device)
    
    ckpt_path = os.path.join(config["paths"]["checkpoint_dir"], "checkpoint_best.pth")
    print(f"[INFO] Loading original model checkpoint: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=device)
    
    # Strip 'module.' prefix if it exists
    state_dict = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint
    new_state_dict = {k[7:] if k.startswith("module.") else k: v for k, v in state_dict.items()}
    model.load_state_dict(new_state_dict, strict=False)
    
    print("[INFO] Model loaded successfully.")
    
    # Fine-tuning uses a much lower learning rate
    lr = 1e-5
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = FocalLoss(alpha=class_weights, gamma=2.0)
    scaler = GradScaler('cuda')
    
    epochs = 10 # Train for 10 epochs on heavily augmented dataset
    best_val_loss = float('inf')
    
    print("\n" + "="*50)
    print("  STARING ROBUST FINE-TUNING")
    print("="*50)
    
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        
        # Training Loop
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs} [Train]", leave=False)
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            
            with autocast('cuda', enabled=True):
                logits = model(images)
                loss = criterion(logits, labels)
                
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
            
        avg_train_loss = train_loss / len(train_loader)
        
        # Validation Loop
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc=f"Epoch {epoch}/{epochs} [Val]", leave=False):
                images, labels = images.to(device), labels.to(device)
                with autocast('cuda', enabled=True):
                    logits = model(images)
                    loss = criterion(logits, labels)
                    
                val_loss += loss.item()
                preds = logits.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
                
        avg_val_loss = val_loss / len(val_loader)
        val_acc = correct / total
        
        print(f"Epoch {epoch}/{epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc*100:.2f}%")
        
        # Only save if val loss improves
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            save_path = os.path.join(config["paths"]["checkpoint_dir"], "checkpoint_robust_best.pth")
            save_checkpoint(model, optimizer, None, epoch, val_acc, avg_val_loss, save_path, is_best=False, scaler=scaler)
            print(f"  --> Saved new best robust model! (Val Loss: {avg_val_loss:.4f})")

if __name__ == "__main__":
    main()
