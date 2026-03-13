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
from dotenv import load_dotenv

# Load .env for OPENAI_API_KEY
load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))

# Add src to path for absolute imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../src")))

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
        self.yolo_model = None
        self.fracture_type_model = None
        self.fracture_type_classes = None
        self.load_error = None
        self.used_swin_fallback = False

    # ── Model loading ─────────────────────────────────────────────────────────

    def _build_classifier_model(self):
        try:
            self.used_swin_fallback = False
            return build_model(self.config).to(self.device)
        except Exception as exc:
            swin_cfg = self.config.get("model", {}).get("swin", {})
            if not swin_cfg.get("pretrained", False):
                raise
            print(f"[WARN] Pretrained Swin init failed: {exc}. Retrying without pretrained weights.")
            self.config["model"]["swin"]["pretrained"] = False
            self.used_swin_fallback = True
            return build_model(self.config).to(self.device)

    def load_model(self):
        self.load_error = None
        self.model = None
        self.grad_cam = None
        try:
            self.config = load_config(self.config_path)
            self.device = get_device()
            self.class_names = self.config["data"]["class_names"]

            self.model = self._build_classifier_model()

            ckpt_dir = os.path.abspath(os.path.join(
                os.path.dirname(__file__), "../../",
                self.config["paths"].get("checkpoint_dir", "checkpoints")
            ))
            ckpt_path = os.path.join(ckpt_dir, "checkpoint_robust_best.pth")
            if not os.path.exists(ckpt_path):
                ckpt_path = os.path.join(ckpt_dir, "checkpoint.pth")
            if not os.path.exists(ckpt_path):
                raise FileNotFoundError(f"No classifier checkpoint found in {ckpt_dir}")

            print(f"Loading weights from {ckpt_path}")
            checkpoint = torch.load(ckpt_path, map_location=self.device)
            state_dict = checkpoint.get("model_state_dict", checkpoint)
            new_state_dict = {(k[7:] if k.startswith("module.") else k): v for k, v in state_dict.items()}
            self.model.load_state_dict(new_state_dict, strict=False)
            self.model.eval()

            self.transform = get_val_transforms(self.config)
            self.grad_cam = GradCAM(self.model)
            print("FractureMamba-ViT model loaded and ready.")

            self.load_yolo()
            self.load_fracture_type_model()

        except Exception as exc:
            self.load_error = str(exc)
            self.model = None
            self.grad_cam = None
            raise

    def load_yolo(self):
        candidate_paths = [
            "../../runs/detect/runs/detect/fracture_yolo3/weights/best.pt",
        ]
        for rel in candidate_paths:
            p = os.path.abspath(os.path.join(os.path.dirname(__file__), rel))
            if os.path.exists(p):
                try:
                    from ultralytics import YOLO
                    self.yolo_model = YOLO(p)
                    print(f"[YOLO] Loaded from {p}")
                except Exception as e:
                    print(f"[YOLO] Failed: {e}")
                return
        print("[YOLO] No model found — localization skipped")

    def load_fracture_type_model(self):
        candidate_paths = [
            "../../runs/fracture_type/best.pth",
            "../../../runs/fracture_type/best.pth",
        ]
        for rel in candidate_paths:
            p = os.path.abspath(os.path.join(os.path.dirname(__file__), rel))
            if os.path.exists(p):
                try:
                    import timm
                    import torch.nn as nn

                    ckpt = torch.load(p, map_location=self.device)
                    classes = ckpt.get("class_names", [])
                    cfg = ckpt.get("config", {}) or {}
                    mc = cfg.get("model", {}) if cfg else {}
                    model_name = mc.get("name", "swin_tiny_patch4_window7_224")
                    drop_path = mc.get("drop_path_rate", 0.2)
                    head_cfg = mc.get("head", {"hidden_dim": 256, "dropout": 0.35})
                    num_classes = len(classes)

                    backbone = timm.create_model(model_name, pretrained=False,
                                                 num_classes=0, drop_path_rate=drop_path)
                    feat_dim = backbone.num_features
                    hidden = head_cfg.get("hidden_dim", 256)
                    drop = head_cfg.get("dropout", 0.35)

                    class _SwinHead(nn.Module):
                        def __init__(self):
                            super().__init__()
                            self.backbone = backbone
                            self.head = nn.Sequential(
                                nn.LayerNorm(feat_dim),
                                nn.Linear(feat_dim, hidden),
                                nn.GELU(),
                                nn.Dropout(drop),
                                nn.Linear(hidden, num_classes),
                            )
                        def forward(self, x):
                            return self.head(self.backbone(x))

                    model = _SwinHead().to(self.device)
                    model.load_state_dict(ckpt["model_state_dict"])
                    model.eval()
                    self.fracture_type_model = model
                    self.fracture_type_classes = classes
                    print(f"[Layer3] Swin fracture type model loaded: {num_classes} classes")
                except Exception as e:
                    print(f"[Layer3] Failed: {e}")
                return
        print("[Layer3] No Swin checkpoint found — Swin type classification skipped")

    # ── Inference helpers ─────────────────────────────────────────────────────

    def predict_fracture_type_swin(self, pil_image):
        """Run Swin-Tiny to predict fracture type. Returns (type, confidence, probs_dict)."""
        try:
            from torchvision import transforms
            tf = transforms.Compose([
                transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ])
            img_tensor = tf(pil_image).unsqueeze(0).to(self.device)
            with torch.no_grad():
                logits = self.fracture_type_model(img_tensor)
                probs = F.softmax(logits, dim=-1).squeeze(0).cpu().numpy()
            top_idx = int(np.argmax(probs))
            top_conf = float(probs[top_idx])
            top_class = self.fracture_type_classes[top_idx]
            top5 = sorted(zip(self.fracture_type_classes, probs.tolist()), key=lambda x: -x[1])[:5]
            print(f"[Swin] Type: {top_class} ({top_conf:.3f})")
            return top_class, top_conf, {c: round(p, 4) for c, p in top5}
        except Exception as e:
            print(f"[Swin] Inference error: {e}")
            return None, None, None

    def predict_fracture_type_openai(self, pil_image, known_classes=None):
        """
        Call gpt-4o-mini to classify fracture type.
        Only called when ensemble has already confirmed fracture.
        Returns: (fracture_type, confidence, None) or (None, None, None)
        """
        try:
            import openai as _openai
            import json as _json

            api_key = os.environ.get("OPENAI_API_KEY", "")
            if not api_key:
                print("[OpenAI] OPENAI_API_KEY not set — skipping")
                return None, None, None

            client = _openai.OpenAI(api_key=api_key)

            # Encode as 512px JPEG for cost efficiency
            buf = io.BytesIO()
            img_resized = pil_image.copy()
            img_resized.thumbnail((512, 512))
            img_resized.save(buf, format="JPEG", quality=85)
            b64_img = base64.b64encode(buf.getvalue()).decode("utf-8")

            # Focused prompt — ensemble already confirmed fracture, just classify type
            prompt = (
                "You are an orthopedic radiologist AI.\n\n"
                "This X-ray has already been confirmed to contain a bone fracture.\n"
                "Your task is ONLY to classify the fracture type based on the fracture line pattern and bone structure.\n\n"
                "Respond in JSON format with ONLY these two fields:\n"
                "{\n"
                "  \"fracture_type\": \"<fracture type name>\",\n"
                "  \"confidence\": <0.0 to 1.0>\n"
                "}\n\n"
                "Do not add any other fields. Do not question whether a fracture exists."
            )

            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {
                            "url": f"data:image/jpeg;base64,{b64_img}",
                            "detail": "low"
                        }}
                    ]
                }],
                max_tokens=80,
                temperature=0.1,
            )

            raw = response.choices[0].message.content.strip()
            print(f"[OpenAI] Raw: {raw}")
            clean = raw.replace("```json", "").replace("```", "").strip()
            parsed = _json.loads(clean)

            fracture_type = parsed.get("fracture_type", "").strip()
            confidence = float(parsed.get("confidence", 0.8))

            # Sanity check — reject non-fracture responses
            if not fracture_type or fracture_type.lower() in ("no fracture", "none", "unknown", "n/a", ""):
                print(f"[OpenAI] Returned invalid type '{fracture_type}' — skipping")
                return None, None, None

            # Fuzzy-match to known class list if provided
            if known_classes:
                matched = next((c for c in known_classes if c.lower() == fracture_type.lower()), None)
                if not matched:
                    matched = next(
                        (c for c in known_classes
                         if fracture_type.lower() in c.lower() or c.lower() in fracture_type.lower()),
                        fracture_type
                    )
                fracture_type = matched

            print(f"[OpenAI] Type: {fracture_type} ({confidence:.2f})")
            return fracture_type, confidence, None

        except Exception as e:
            print(f"[OpenAI] Failed: {e}")
            return None, None, None

    def _ensemble_fracture_type(self, pil_image):
        """Run Swin + OpenAI and return final fracture type decision."""
        known_cls = self.fracture_type_classes if self.fracture_type_classes else None

        swin_type = swin_conf = swin_probs = None
        if self.fracture_type_model is not None:
            swin_type, swin_conf, swin_probs = self.predict_fracture_type_swin(pil_image)

        oai_type, oai_conf, _ = self.predict_fracture_type_openai(pil_image, known_classes=known_cls)

        fracture_type = fracture_type_conf = fracture_type_probs = None
        openai_override = False
        openai_reasoning = None

        if swin_type and oai_type:
            if swin_type.lower() == oai_type.lower():
                fracture_type = swin_type
                fracture_type_conf = swin_conf
                fracture_type_probs = swin_probs
                openai_override = False
                print(f"[L3] Agree: {swin_type}")
            else:
                fracture_type = oai_type
                fracture_type_conf = oai_conf
                fracture_type_probs = swin_probs
                openai_override = True
                print(f"[L3] Disagree: Swin={swin_type}, OpenAI={oai_type} -> OpenAI wins")
        elif oai_type:
            fracture_type = oai_type
            fracture_type_conf = oai_conf
            openai_override = True
            print(f"[L3] OpenAI only: {oai_type}")
        elif swin_type:
            fracture_type = swin_type
            fracture_type_conf = swin_conf
            fracture_type_probs = swin_probs
            print(f"[L3] Swin only: {swin_type}")

        return {
            "fracture_type": fracture_type,
            "fracture_type_confidence": fracture_type_conf,
            "fracture_type_probabilities": fracture_type_probs,
            "openai_override": openai_override,
            "openai_reasoning": openai_reasoning,
        }

    # ── Utility ───────────────────────────────────────────────────────────────

    def image_to_base64(self, img_array):
        """Convert numpy RGB image to base64 PNG string."""
        is_success, buffer = cv2.imencode(".png", cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR))
        return base64.b64encode(buffer).decode("utf-8")

    def fig_to_base64(self, fig):
        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", dpi=100)
        buf.seek(0)
        encoded = base64.b64encode(buf.read()).decode("utf-8")
        plt.close(fig)
        return encoded

    # ── YOLO localization ─────────────────────────────────────────────────────

    def run_yolo(self, pil_image):
        """Run YOLO on PIL image. Returns (yolo_image_b64, bbox_dict, detected, fracture_location)."""
        if self.yolo_model is None:
            return None, None, False, "unknown"
        try:
            img_rgb = np.array(pil_image)
            results = self.yolo_model(img_rgb, verbose=False)
            boxes = results[0].boxes

            if len(boxes) == 0:
                return None, None, False, "unknown"

            confs = boxes.conf.cpu().numpy()
            best = int(np.argmax(confs))
            xyxy = boxes.xyxy[best].cpu().numpy()
            conf = float(confs[best])
            x1, y1, x2, y2 = map(int, xyxy)
            h, w = img_rgb.shape[:2]

            bbox = {"x1": x1, "y1": y1, "x2": x2, "y2": y2, "conf": round(conf, 3)}

            cy = (y1 + y2) / 2 / h
            if cy < 0.33:
                fracture_location = "upper region"
            elif cy < 0.66:
                fracture_location = "mid-shaft"
            else:
                fracture_location = "lower region"

            annotated = img_rgb.copy()
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 120, 255), 3)
            cv2.putText(annotated, f"fractured {conf:.2f}", (x1, max(y1 - 10, 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 120, 255), 2)
            yolo_image_b64 = self.image_to_base64(annotated)

            return yolo_image_b64, bbox, True, fracture_location

        except Exception as e:
            print(f"[YOLO] Error: {e}")
            return None, None, False, "unknown"

    # ── Visualizations ────────────────────────────────────────────────────────

    def generate_gradcam(self, img_tensor, orig_img_rgb, pred_class_idx):
        """Generate GradCAM heatmap overlaid on original image."""
        try:
            cam = self.grad_cam.generate(img_tensor, target_class=pred_class_idx)
            cam_resized = cv2.resize(cam, (orig_img_rgb.shape[1], orig_img_rgb.shape[0]))
            cam_norm = (cam_resized - cam_resized.min()) / (cam_resized.max() - cam_resized.min() + 1e-8)
            heatmap = (cm.jet(cam_norm)[:, :, :3] * 255).astype(np.uint8)
            overlay = cv2.addWeighted(orig_img_rgb, 0.55, heatmap, 0.45, 0)
            return self.image_to_base64(overlay)
        except Exception as e:
            print(f"[GradCAM] Error: {e}")
            return self.image_to_base64(orig_img_rgb)

    def generate_mamba_viz(self, img_tensor, orig_img_rgb):
        """Attempt to generate Mamba state visualisation (best-effort)."""
        try:
            with torch.no_grad():
                features = self.model.get_features(img_tensor)
            if features is None:
                return None
            feat = features[0].mean(0).cpu().numpy()
            feat_resized = cv2.resize(feat, (orig_img_rgb.shape[1], orig_img_rgb.shape[0]))
            feat_norm = (feat_resized - feat_resized.min()) / (feat_resized.max() - feat_resized.min() + 1e-8)
            heatmap = (cm.plasma(feat_norm)[:, :, :3] * 255).astype(np.uint8)
            overlay = cv2.addWeighted(orig_img_rgb, 0.6, heatmap, 0.4, 0)
            return self.image_to_base64(overlay)
        except Exception:
            return None

    # ── Main pipeline ─────────────────────────────────────────────────────────

    def predict_and_explain(self, pil_image):
        if self.model is None:
            self.load_model()
        if self.model is None:
            raise RuntimeError(self.load_error or "Classifier model is not loaded")

        orig_img_rgb = np.array(pil_image)

        # Layer 1: ViT fracture detection
        transformed = self.transform(image=orig_img_rgb)
        img_tensor = transformed["image"].unsqueeze(0).to(self.device)

        with torch.no_grad():
            with torch.amp.autocast("cuda", enabled=(self.device.type == "cuda")):
                logits = self.model(img_tensor)
                probs = F.softmax(logits, dim=1)[0].cpu().numpy()

        pred_class_idx = int(np.argmax(probs))
        pred_class_name = self.class_names[pred_class_idx]
        confidence = float(probs[pred_class_idx])
        all_probs = {name: round(float(p), 4) for name, p in zip(self.class_names, probs)}

        vit_says_fracture = "not" not in pred_class_name.lower()
        print(f"[ViT] {pred_class_name} ({confidence:.3f}), fracture={vit_says_fracture}")

        # GradCAM (always generated)
        gradcam_b64 = self.generate_gradcam(img_tensor, orig_img_rgb, pred_class_idx)
        mamba_b64 = self.generate_mamba_viz(img_tensor, orig_img_rgb)

        # Layer 2: YOLO localization (always run)
        yolo_image_b64, bbox, yolo_detected, fracture_location = self.run_yolo(pil_image)

        # Ensemble verdict
        # Case 1: ViT=fracture, YOLO=box   → Confirmed, show YOLO
        # Case 2: ViT=fracture, YOLO=none  → Confirmed, show GradCAM
        # Case 3: ViT=none,     YOLO=box   → Override to fractured, show YOLO
        # Case 4: ViT=none,     YOLO=none  → Not fractured
        if vit_says_fracture and yolo_detected:
            ensemble_verdict = "fractured"
            localization_status = "success"
            print("[Ensemble] ViT=fracture, YOLO=box -> Confirmed")

        elif vit_says_fracture and not yolo_detected:
            ensemble_verdict = "fractured"
            localization_status = "failed"
            yolo_image_b64 = None
            print("[Ensemble] ViT=fracture, YOLO=none -> Fracture confirmed, localization failed")

        elif not vit_says_fracture and yolo_detected:
            ensemble_verdict = "fractured"
            localization_status = "success"
            fractured_class = next(
                (c for c in self.class_names if "not" not in c.lower() and "fracture" in c.lower()),
                "fractured"
            )
            pred_class_name = fractured_class
            confidence = max(confidence, bbox["conf"])
            print(f"[Ensemble] ViT=none, YOLO=box -> OVERRIDE (YOLO conf={bbox['conf']:.3f})")

        else:
            ensemble_verdict = "not_fractured"
            localization_status = "not_applicable"
            yolo_image_b64 = None
            fracture_location = None
            print("[Ensemble] ViT=none, YOLO=none -> Not fractured")

        # Layer 3: Fracture type (only when fractured)
        type_result = {
            "fracture_type": None,
            "fracture_type_confidence": None,
            "fracture_type_probabilities": None,
            "openai_override": False,
            "openai_reasoning": None,
        }
        if ensemble_verdict == "fractured":
            type_result = self._ensemble_fracture_type(pil_image)

        return {
            "prediction": pred_class_name,
            "confidence": confidence,
            "probabilities": all_probs,
            "gradcam_base64": f"data:image/png;base64,{gradcam_b64}",
            "mamba_base64": f"data:image/png;base64,{mamba_b64}" if mamba_b64 else None,
            "fracture_location": fracture_location,
            "bbox": bbox,
            "yolo_image_base64": f"data:image/png;base64,{yolo_image_b64}" if yolo_image_b64 else None,
            "ensemble_verdict": ensemble_verdict,
            "localization_status": localization_status,
            **type_result,
        }
