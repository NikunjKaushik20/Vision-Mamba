import os
import sys
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))
from utils import load_config, get_device, load_checkpoint
from data_loader import get_val_transforms
from model import build_model
import torch.nn.functional as F

def get_gradcam(model, image_tensor, target_class):
    # Register hooks for the final feature map of Swin and Mamba
    # We will just hook the final fusion output right before the classifier for simplicity
    
    features = []
    gradients = []
    
    def fw_hook(module, input, output):
        features.append(output)
        
    def bw_hook(module, grad_in, grad_out):
        gradients.append(grad_out[0])
        
    # Hook the fusion layer
    handle_fw = model.fusion.register_forward_hook(fw_hook)
    handle_bw = model.fusion.register_backward_hook(bw_hook)
    
    model.eval()
    
    # Forward pass
    logits = model(image_tensor)
    
    model.zero_grad()
    # Backward pass for target class
    loss = logits[0, target_class]
    loss.backward()
    
    handle_fw.remove()
    handle_bw.remove()
    
    # The fusion layer outputs shape (B, dim), not a spatial map.
    # So we need to hook the Swin stream output which actually has spatial dimensions.
    return None

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--img", required=True, help="Path to image")
    args = parser.parse_args()
    
    config = load_config("d:/Vision-Mamba/config.yaml")
    device = get_device()
    model = build_model(config).to(device)
    
    # Load checkpoint
    ckpt_path = os.path.join(config["paths"]["checkpoint_dir"], "checkpoint_best.pth")
    checkpoint = torch.load(ckpt_path, map_location=device)
    state_dict = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint
    
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v
    model.load_state_dict(new_state_dict, strict=False)
    
    model.eval()
    
    transform = get_val_transforms(config)
    class_names = config["data"]["class_names"]
    
    img_path = args.img
    orig_img = cv2.imread(img_path)
    if orig_img is None:
        print("Image not found")
        return
        
    orig_img_rgb = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)
    transformed = transform(image=orig_img_rgb)
    img_tensor = transformed["image"].unsqueeze(0).to(device)
    
    with torch.no_grad():
        with torch.amp.autocast('cuda', enabled=True):
            logits = model(img_tensor)
            probs = F.softmax(logits, dim=1)[0]
            
    print("\n--- PREDICTION ---")
    for i, name in enumerate(class_names):
        print(f"{name}: {probs[i].item()*100:.2f}%")
        
if __name__ == "__main__":
    main()
