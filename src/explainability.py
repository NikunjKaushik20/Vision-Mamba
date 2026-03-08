
import os, sys
sys.path.insert(0, os.path.dirname(__file__))
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import cv2
from torch.cuda.amp import autocast
from tqdm import tqdm

from utils import load_config, seed_everything, get_device, load_checkpoint, ensure_dirs
from data_loader import BoneFractureDataset, get_val_transforms
from model import build_model


class GradCAM:
    """
    Grad-CAM implementation for the Swin Transformer stream.
    Generates class-specific saliency maps highlighting fracture regions.
    """
    
    def __init__(self, model, target_layer=None):
        self.model = model
        self.gradients = None
        self.activations = None
        
        # Hook the last layer of Swin Transformer
        if target_layer is None:
            # Use the last norm layer of Swin
            target_layer = model.swin_stream.model.norm
        
        self.target_layer = target_layer
        self._register_hooks()
    
    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()
        
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()
        
        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)
    
    def generate(self, input_tensor, target_class=None):
        """
        Generate Grad-CAM heatmap.
        
        Args:
            input_tensor: (1, 3, H, W) - input image
            target_class: Target class index. If None, uses predicted class.
        
        Returns:
            heatmap: (H, W) numpy array with values in [0, 1]
        """
        self.model.eval()
        
        # Enable gradients for this
        input_tensor.requires_grad_(True)
        
        # Forward
        output = self.model(input_tensor)
        
        if target_class is None:
            target_class = output.argmax(dim=1).item()
        
        # Backward for target class
        self.model.zero_grad()
        score = output[0, target_class]
        score.backward(retain_graph=True)
        
        if self.gradients is None or self.activations is None:
            print("[WARN] Grad-CAM: No gradients/activations captured. Returning zero map.")
            return np.zeros((input_tensor.shape[2], input_tensor.shape[3]))
        
        # Compute weights
        weights = self.gradients.mean(dim=(0, 1) if self.gradients.dim() == 3 else (2, 3))
        
        if self.activations.dim() == 3:
            # (B, seq_len, dim) from transformer
            act = self.activations[0]  # (seq_len, dim)
            
            if weights.dim() == 1:
                cam = (act * weights.unsqueeze(0)).sum(-1)  # (seq_len,)
            else:
                cam = (act * weights).sum(-1)
            
            # Reshape to spatial
            h = w = int(np.sqrt(cam.shape[0]))
            if h * w != cam.shape[0]:
                # Pad or truncate
                total = h * w
                if cam.shape[0] > total:
                    cam = cam[:total]
                else:
                    cam = F.pad(cam, (0, total - cam.shape[0]))
                h = w = int(np.sqrt(cam.shape[0]))
            
            cam = cam.reshape(h, w)
        else:
            # (B, C, H, W) from CNN
            cam = (weights.unsqueeze(-1).unsqueeze(-1) * self.activations[0]).sum(0)
        
        # ReLU and normalize
        cam = F.relu(cam)
        cam = cam.detach().cpu().numpy()
        
        if cam.max() > 0:
            cam = cam / cam.max()
        
        # Resize to input size — guard against 0-size arrays
        cam = np.float32(cam)
        H, W = input_tensor.shape[2], input_tensor.shape[3]
        if cam.size == 0 or cam.shape[0] == 0 or cam.shape[1] == 0:
            print("[WARN] Grad-CAM: cam is empty, returning zero map.")
            return np.zeros((H, W), dtype=np.float32)
        cam = cv2.resize(cam, (W, H))
        
        return cam


def generate_attention_maps(model, images, save_dir, class_names, device, num_samples=5):
    """
    Generate and save attention visualizations from the Mamba stream.
    Visualizes how the model attends to different image regions.
    """
    os.makedirs(save_dir, exist_ok=True)
    model.eval()
    
    print("\n[EXPLAIN] Generating attention maps...")
    
    for idx in range(min(num_samples, len(images))):
        img_tensor = images[idx].unsqueeze(0).to(device)
        
        with torch.no_grad():
            logits = model(img_tensor)
            pred_class = logits.argmax(dim=1).item()
            
            # Get Mamba tokens
            mamba_cls, mamba_tokens = model.mamba_stream(img_tensor)
        
        # Compute attention from token norms (proxy for importance)
        token_importance = mamba_tokens[0].norm(dim=-1).cpu().numpy()  # (num_patches,)
        
        # Reshape to spatial grid
        h = w = int(np.sqrt(len(token_importance)))
        if h * w == len(token_importance):
            attention_map = token_importance.reshape(h, w)
        else:
            h = w = int(np.ceil(np.sqrt(len(token_importance))))
            padded = np.zeros(h * w)
            padded[:len(token_importance)] = token_importance
            attention_map = padded.reshape(h, w)
        
        # Normalize
        if attention_map.max() > attention_map.min():
            attention_map = (attention_map - attention_map.min()) / (attention_map.max() - attention_map.min())
        
        # Upsample to image size
        attention_map = cv2.resize(attention_map, (224, 224), interpolation=cv2.INTER_CUBIC)
        
        # Denormalize image for display
        img_display = denormalize_image(images[idx])
        
        # Plot
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        axes[0].imshow(img_display)
        axes[0].set_title("Original X-ray", fontsize=12)
        axes[0].axis("off")
        
        axes[1].imshow(attention_map, cmap="jet")
        axes[1].set_title("Mamba Attention Map", fontsize=12)
        axes[1].axis("off")
        
        # Overlay
        heatmap_colored = cm.jet(attention_map)[:, :, :3]
        overlay = 0.5 * img_display / 255.0 + 0.5 * heatmap_colored
        overlay = np.clip(overlay, 0, 1)
        axes[2].imshow(overlay)
        axes[2].set_title(f"Overlay (Pred: {class_names[pred_class]})", fontsize=12)
        axes[2].axis("off")
        
        plt.suptitle("Vision Mamba Attention Visualization", fontsize=14, fontweight="bold")
        plt.tight_layout()
        
        save_path = os.path.join(save_dir, f"attention_map_{idx+1}.png")
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        
        print(f"  Saved attention map {idx+1}: {save_path}")


def generate_gradcam(model, images, labels, save_dir, class_names, device, num_samples=5):
    """
    Generate and save Grad-CAM visualizations.
    Shows which image regions drive the classification decision.
    """
    os.makedirs(save_dir, exist_ok=True)
    
    print("\n[EXPLAIN] Generating Grad-CAM saliency maps...")
    
    grad_cam = GradCAM(model)
    
    for idx in range(min(num_samples, len(images))):
        try:
            img_tensor = images[idx].unsqueeze(0).to(device)
            true_label = labels[idx] if labels is not None else None
            
            # Recreate GradCAM each sample to reset hooks
            grad_cam = GradCAM(model)

            # Get prediction
            with torch.no_grad():
                logits = model(img_tensor)
                pred_class = logits.argmax(dim=1).item()
                pred_prob = F.softmax(logits, dim=1)[0, pred_class].item()
            
            # Generate Grad-CAM
            cam = grad_cam.generate(img_tensor.clone().detach().requires_grad_(True), target_class=pred_class)
            
            # Denormalize image
            img_display = denormalize_image(images[idx])
            
            # Plot
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            
            axes[0].imshow(img_display)
            true_str = f" (True: {class_names[true_label]})" if true_label is not None else ""
            axes[0].set_title(f"Original X-ray{true_str}", fontsize=11)
            axes[0].axis("off")
            
            axes[1].imshow(cam, cmap="jet")
            axes[1].set_title("Grad-CAM Heatmap", fontsize=12)
            axes[1].axis("off")
            
            # Overlay
            heatmap_colored = cm.jet(cam)[:, :, :3]
            overlay = 0.5 * img_display / 255.0 + 0.5 * heatmap_colored
            overlay = np.clip(overlay, 0, 1)
            axes[2].imshow(overlay)
            axes[2].set_title(f"Pred: {class_names[pred_class]} ({pred_prob*100:.1f}%)", fontsize=12)
            axes[2].axis("off")
            
            plt.suptitle("Grad-CAM Saliency Visualization", fontsize=14, fontweight="bold")
            plt.tight_layout()
            
            save_path = os.path.join(save_dir, f"gradcam_{idx+1}.png")
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            plt.close()
            
            print(f"  Saved Grad-CAM {idx+1}: {save_path}")
        except Exception as e:
            print(f"  [WARN] Grad-CAM {idx+1} failed: {e}. Skipping.")
            plt.close('all')
            continue


def generate_mamba_state_viz(model, images, save_dir, class_names, device, num_samples=3):
    """
    Visualize Mamba hidden state evolution.
    Shows how the SSM state changes across the image patch sequence —
    useful for understanding what sequential patterns the model detects.
    """
    os.makedirs(save_dir, exist_ok=True)
    
    print("\n[EXPLAIN] Generating Mamba state visualizations...")
    
    model.eval()
    
    for idx in range(min(num_samples, len(images))):
        img_tensor = images[idx].unsqueeze(0).to(device)
        
        with torch.no_grad():
            # Get Mamba tokens (sequential features)
            mamba_cls, mamba_tokens = model.mamba_stream(img_tensor)
            logits = model(img_tensor)
            pred_class = logits.argmax(dim=1).item()
        
        # Token features evolution
        tokens_np = mamba_tokens[0].cpu().numpy()  # (num_patches, embed_dim)
        
        # Denormalize image
        img_display = denormalize_image(images[idx])
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        
        # Original image
        axes[0, 0].imshow(img_display)
        axes[0, 0].set_title(f"Input X-ray (Pred: {class_names[pred_class]})", fontsize=12)
        axes[0, 0].axis("off")
        
        # Token feature heatmap (first 64 dimensions)
        show_dims = min(64, tokens_np.shape[1])
        axes[0, 1].imshow(tokens_np[:, :show_dims].T, aspect="auto", cmap="viridis")
        axes[0, 1].set_xlabel("Patch Position", fontsize=10)
        axes[0, 1].set_ylabel("Feature Dimension", fontsize=10)
        axes[0, 1].set_title("Mamba Hidden State Evolution", fontsize=12)
        
        # Token norms (importance over sequence)
        norms = np.linalg.norm(tokens_np, axis=1)
        axes[1, 0].plot(norms, "b-", linewidth=1.5)
        axes[1, 0].fill_between(range(len(norms)), norms, alpha=0.3, color="blue")
        axes[1, 0].set_xlabel("Patch Position", fontsize=10)
        axes[1, 0].set_ylabel("Feature Norm", fontsize=10)
        axes[1, 0].set_title("Token Importance (L2 Norm)", fontsize=12)
        axes[1, 0].grid(True, alpha=0.3)
        
        # Cosine similarity between adjacent tokens
        if tokens_np.shape[0] > 1:
            cos_sims = []
            for i in range(len(tokens_np) - 1):
                a, b = tokens_np[i], tokens_np[i+1]
                cos = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)
                cos_sims.append(cos)
            axes[1, 1].plot(cos_sims, "r-", linewidth=1.5)
            axes[1, 1].fill_between(range(len(cos_sims)), cos_sims, alpha=0.3, color="red")
            axes[1, 1].set_xlabel("Adjacent Patch Pair", fontsize=10)
            axes[1, 1].set_ylabel("Cosine Similarity", fontsize=10)
            axes[1, 1].set_title("Sequential Coherence", fontsize=12)
            axes[1, 1].grid(True, alpha=0.3)
            axes[1, 1].set_ylim(-0.2, 1.1)
        
        plt.suptitle("Mamba State Space Visualization", fontsize=15, fontweight="bold")
        plt.tight_layout()
        
        save_path = os.path.join(save_dir, f"mamba_state_{idx+1}.png")
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        
        print(f"  Saved Mamba state viz {idx+1}: {save_path}")


def generate_comparison_grid(model, images, labels, save_dir, class_names, device, num_samples=8):
    """
    Generate a comparison grid showing predictions vs ground truth.
    Useful for the presentation to show model performance at a glance.
    """
    os.makedirs(save_dir, exist_ok=True)
    
    print("\n[EXPLAIN] Generating prediction comparison grid...")
    model.eval()
    
    n = min(num_samples, len(images))
    cols = 4
    rows = (n + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    if rows == 1:
        axes = axes.reshape(1, -1)
    
    for idx in range(n):
        r, c = idx // cols, idx % cols
        
        img_tensor = images[idx].unsqueeze(0).to(device)
        
        with torch.no_grad():
            logits = model(img_tensor)
            pred = logits.argmax(dim=1).item()
            prob = F.softmax(logits, dim=1)[0, pred].item()
        
        img_display = denormalize_image(images[idx])
        true_label = labels[idx] if labels is not None else None
        
        axes[r, c].imshow(img_display)
        
        correct = (true_label == pred) if true_label is not None else None
        color = "green" if correct else "red"
        
        title = f"Pred: {class_names[pred]} ({prob*100:.0f}%)"
        if true_label is not None:
            title += f"\nTrue: {class_names[true_label]}"
        
        axes[r, c].set_title(title, fontsize=9, color=color, fontweight="bold")
        axes[r, c].axis("off")
    
    # Hide unused axes
    for idx in range(n, rows * cols):
        r, c = idx // cols, idx % cols
        axes[r, c].axis("off")
    
    plt.suptitle("FractureMamba-ViT Predictions", fontsize=16, fontweight="bold")
    plt.tight_layout()
    
    save_path = os.path.join(save_dir, "prediction_grid.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    
    print(f"  Saved prediction grid: {save_path}")


def denormalize_image(tensor):
    """Denormalize a tensor to display as an image."""
    mean = torch.tensor([0.485, 0.456, 0.406])
    std = torch.tensor([0.229, 0.224, 0.225])
    
    img = tensor.cpu().clone()
    for t, m, s in zip(img, mean, std):
        t.mul_(s).add_(m)
    
    img = img.permute(1, 2, 0).numpy()
    img = np.clip(img * 255, 0, 255).astype(np.uint8)
    return img


def main():
    parser = argparse.ArgumentParser(description="Generate explainability visualizations")
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--num-samples", type=int, default=5)
    args = parser.parse_args()
    
    config = load_config(args.config)
    seed_everything(config["seed"])
    device = get_device()
    ensure_dirs(config)
    
    # Load data
    val_transform = get_val_transforms(config)
    test_dataset = BoneFractureDataset(
        config["data"]["root_dir"], split="test",
        transform=val_transform, config=config
    )
    
    class_names = test_dataset.classes
    config["data"]["num_classes"] = len(class_names)
    
    # Load model
    model = build_model(config).to(device)
    
    if args.checkpoint:
        load_checkpoint(model, args.checkpoint)
    else:
        ckpt_dir = config["paths"]["checkpoint_dir"]
        for name in ["checkpoint_swa.pth", "checkpoint_best.pth", "checkpoint.pth"]:
            path = os.path.join(ckpt_dir, name)
            if os.path.exists(path):
                load_checkpoint(model, path)
                break
    
    # Get sample images
    n = args.num_samples
    images = []
    labels = []
    for i in range(min(n * 2, len(test_dataset))):
        img, lbl = test_dataset[i]
        images.append(img)
        labels.append(lbl.item() if isinstance(lbl, torch.Tensor) else lbl)
    
    images_tensor = torch.stack(images[:n])
    labels_array = labels[:n]
    
    save_dir = config["paths"]["explainability_dir"]
    
    # Generate all visualizations
    generate_attention_maps(model, images_tensor, save_dir, class_names, device, n)
    generate_gradcam(model, images_tensor, labels_array, save_dir, class_names, device, n)
    generate_mamba_state_viz(model, images_tensor, save_dir, class_names, device, min(n, 3))
    
    # Get more samples for the prediction grid
    grid_images = torch.stack(images[:min(8, len(images))])
    grid_labels = labels[:min(8, len(labels))]
    generate_comparison_grid(model, grid_images, grid_labels, save_dir, class_names, device, 8)
    
    print(f"\n[DONE] All explainability outputs saved to {save_dir}")


if __name__ == "__main__":
    main()
