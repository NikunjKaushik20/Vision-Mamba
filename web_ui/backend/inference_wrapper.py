import os
import sys
import io
import base64
import torch
import torch.nn.functional as F
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# Add src to path for absolute imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))

from utils import load_config, get_device
from data_loader import get_val_transforms
from model import build_model
from explainability import GradCAM

class FractureModelManager:
    def __init__(self, config_path="../../config.yaml"):
        self.config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), config_path))
        self.model = None
        self.device = None
        self.config = None
        self.transform = None
        self.class_names = None
        self.grad_cam = None

    def load_model(self):
        self.config = load_config(self.config_path)
        self.device = get_device()
        self.class_names = self.config["data"]["class_names"]
        
        self.model = build_model(self.config).to(self.device)
        
        # Load weights
        ckpt_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../', self.config["paths"].get("checkpoint_dir", "checkpoints")))
        ckpt_path = os.path.join(ckpt_dir, "checkpoint_robust_best.pth")
        
        if not os.path.exists(ckpt_path):
            print(f"[WARN] Checkpoint not found at {ckpt_path}. Looking for alternative...")
            ckpt_path = os.path.join(ckpt_dir, "checkpoint.pth")
            
        print(f"Loading weights from {ckpt_path}")
        checkpoint = torch.load(ckpt_path, map_location=self.device)
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        
        # Strip module if loaded from DDP
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith("module."):
                new_state_dict[k[7:]] = v
            else:
                new_state_dict[k] = v
                
        self.model.load_state_dict(new_state_dict, strict=False)
        self.model.eval()
        
        self.transform = get_val_transforms(self.config)
        self.grad_cam = GradCAM(self.model)
        print("FractureMamba model loaded and ready.")

    def image_to_base64(self, img_array):
        """Convert numpy image (RGB or BGR) or matplotlib figure to base64 string"""
        is_success, buffer = cv2.imencode(".png", cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR))
        return base64.b64encode(buffer).decode("utf-8")
        
    def fig_to_base64(self, fig):
        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", dpi=100)
        buf.seek(0)
        encoded = base64.b64encode(buf.read()).decode("utf-8")
        plt.close(fig)
        return encoded

    def predict_and_explain(self, pil_image):
        if self.model is None:
            self.load_model()
            
        # Convert PIL to CV2 format (RGB)
        orig_img_rgb = np.array(pil_image)
        
        # Transform
        transformed = self.transform(image=orig_img_rgb)
        img_tensor = transformed["image"].unsqueeze(0).to(self.device)
        
        # 1. Prediction
        with torch.no_grad():
            with torch.amp.autocast('cuda', enabled=True):
                logits = self.model(img_tensor)
                probs = F.softmax(logits, dim=1)[0].cpu().numpy()
                
        pred_class_idx = np.argmax(probs)
        pred_class_name = self.class_names[pred_class_idx]
        confidence = float(probs[pred_class_idx])
        
        # Format probabilities safely
        all_probs = {self.class_names[i]: float(probs[i]) for i in range(len(self.class_names))}
        
        # 2. Grad-CAM — recreate per request so hooks are fresh
        try:
            self.model.zero_grad()
            grad_cam = GradCAM(self.model)
            cam = grad_cam.generate(
                img_tensor.clone().detach().requires_grad_(True),
                target_class=int(pred_class_idx)
            )

            H, W = 224, 224
            # Guard: cam must be non-empty and 2-D with positive dimensions
            if cam is None or cam.ndim != 2 or cam.shape[0] == 0 or cam.shape[1] == 0:
                cam = np.zeros((H, W), dtype=np.float32)

            cam = cv2.resize(cam.astype(np.float32), (W, H))
            heatmap_colored = cm.jet(cam)[:, :, :3]
            img_display = cv2.resize(orig_img_rgb, (W, H))
            overlay = 0.5 * (img_display / 255.0) + 0.5 * heatmap_colored
            overlay = np.clip(overlay, 0, 1)
            gradcam_img_uint8 = (overlay * 255).astype(np.uint8)
            gradcam_b64 = self.image_to_base64(gradcam_img_uint8)
        except Exception as e:
            print(f"[WARN] Grad-CAM failed: {e}. Returning plain scan.")
            img_resized = cv2.resize(orig_img_rgb, (224, 224))
            gradcam_b64 = self.image_to_base64(img_resized)
        
        # 3. Mamba State Visualization
        try:
            with torch.no_grad():
                mamba_cls, mamba_tokens = self.model.mamba_stream(img_tensor)
                
            tokens_np = mamba_tokens[0].cpu().numpy()
            norms = np.linalg.norm(tokens_np, axis=1)
            
            fig, ax = plt.subplots(figsize=(6, 2))
            ax.plot(norms, "b-", linewidth=2)
            ax.fill_between(range(len(norms)), norms, alpha=0.3, color="blue")
            ax.set_title("Mamba Sequential Coherence (Token Norms)", fontsize=10, color='white')
            ax.set_facecolor('#0F172A') # match tailwind dark background
            fig.patch.set_facecolor('#0F172A')
            ax.tick_params(colors='white')
            for spine in ax.spines.values():
                spine.set_color('white')
                
            plt.tight_layout()
            mamba_b64 = self.fig_to_base64(fig)
        except Exception as e:
            print(f"Error generating Mamba viz: {e}")
            mamba_b64 = None

        return {
            "prediction": pred_class_name,
            "confidence": confidence,
            "probabilities": all_probs,
            "gradcam_base64": f"data:image/png;base64,{gradcam_b64}",
            "mamba_base64": f"data:image/png;base64,{mamba_b64}" if mamba_b64 else None
        }
