import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from einops import rearrange

from mamba_module import VisionMamba


class CrossAttentionFusion(nn.Module):

    def __init__(self, dim, num_heads=8, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        # mamba → swin
        self.q_m = nn.Linear(dim, dim)
        self.k_s = nn.Linear(dim, dim)
        self.v_s = nn.Linear(dim, dim)

        # swin → mamba
        self.q_s = nn.Linear(dim, dim)
        self.k_m = nn.Linear(dim, dim)
        self.v_m = nn.Linear(dim, dim)

        self.attn_drop = nn.Dropout(dropout)
        self.gate = nn.Sequential(nn.Linear(dim * 2, dim), nn.Sigmoid())
        self.out_proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(dim * 4, dim), nn.Dropout(dropout),
        )

    def _attn(self, q, k, v):
        B, N, _ = q.shape
        q = rearrange(q, "b n (h d) -> b h n d", h=self.num_heads)
        k = rearrange(k, "b m (h d) -> b h m d", h=self.num_heads)
        v = rearrange(v, "b m (h d) -> b h m d", h=self.num_heads)
        w = self.attn_drop(F.softmax(torch.matmul(q, k.transpose(-2, -1)) * self.scale, dim=-1))
        return rearrange(torch.matmul(w, v), "b h n d -> b n (h d)")

    def forward(self, mamba_f, swin_f):
        ms = mamba_f.unsqueeze(1)
        ss = swin_f.unsqueeze(1)

        m2s = self.norm1(ms + self._attn(self.q_m(ms), self.k_s(ss), self.v_s(ss))).squeeze(1)
        s2m = self.norm2(ss + self._attn(self.q_s(ss), self.k_m(ms), self.v_m(ms))).squeeze(1)

        gate = self.gate(torch.cat([m2s, s2m], dim=-1))
        fused = (gate * m2s + (1 - gate) * s2m).unsqueeze(1)
        fused = (fused + self.ffn(self.norm3(fused))).squeeze(1)
        return self.dropout(self.out_proj(fused))


class SwinFeatureExtractor(nn.Module):
    """Thin wrapper around a timm Swin Transformer — removes the classification head."""

    def __init__(self, model_name="swin_tiny_patch4_window7_224", pretrained=True, drop_path_rate=0.2):
        super().__init__()
        self.model = timm.create_model(model_name, pretrained=pretrained, num_classes=0, drop_path_rate=drop_path_rate)
        self.out_dim = self.model.num_features

    def forward(self, x):
        return self.model(x)


class FractureMambaViT(nn.Module):
    """
    Dual-stream hybrid: Vision Mamba (long-range SSM) + Swin Transformer (local attention),
    fused via bidirectional cross-attention with gating, then classified by a small MLP.
    """

    def __init__(self, num_classes=2, config=None):
        super().__init__()

        if config is None:
            config = {
                "model": {
                    "mamba": {"embed_dim": 192, "depth": 12, "patch_size": 16,
                               "d_state": 16, "d_conv": 4, "expand_factor": 2,
                               "drop_path_rate": 0.2, "dropout": 0.1},
                    "swin": {"model_name": "swin_tiny_patch4_window7_224",
                              "pretrained": True, "drop_path_rate": 0.2},
                    "fusion": {"dim": 384, "num_heads": 8, "dropout": 0.1},
                    "head": {"hidden_dim": 512, "dropout": 0.3},
                }
            }

        mc = config["model"]
        mb, sw, fu, hd = mc["mamba"], mc["swin"], mc["fusion"], mc["head"]

        self.mamba_stream = VisionMamba(
            img_size=config.get("data", {}).get("image_size", 224),
            patch_size=mb["patch_size"], in_channels=3,
            embed_dim=mb["embed_dim"], depth=mb["depth"],
            d_state=mb["d_state"], d_conv=mb["d_conv"],
            expand_factor=mb["expand_factor"], dropout=mb["dropout"],
            drop_path_rate=mb["drop_path_rate"],
        )

        self.swin_stream = SwinFeatureExtractor(sw["model_name"], sw["pretrained"], sw["drop_path_rate"])

        fd = fu["dim"]
        self.mamba_proj = nn.Sequential(nn.Linear(mb["embed_dim"], fd), nn.LayerNorm(fd), nn.GELU())
        self.swin_proj  = nn.Sequential(nn.Linear(self.swin_stream.out_dim, fd), nn.LayerNorm(fd), nn.GELU())
        self.fusion = CrossAttentionFusion(fd, fu["num_heads"], fu["dropout"])

        hdim = hd["hidden_dim"]
        self.classifier = nn.Sequential(
            nn.Linear(fd, hdim), nn.LayerNorm(hdim), nn.GELU(), nn.Dropout(hd["dropout"]),
            nn.Linear(hdim, hdim // 2), nn.GELU(), nn.Dropout(hd["dropout"] * 0.5),
            nn.Linear(hdim // 2, num_classes),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        mamba_cls, mamba_tokens = self.mamba_stream(x)
        swin_feat = self.swin_stream(x)
        fused = self.fusion(self.mamba_proj(mamba_cls), self.swin_proj(swin_feat))
        return self.classifier(fused)

    def get_attention_maps(self, x):
        mamba_cls, mamba_tokens = self.mamba_stream(x)
        return mamba_tokens, self.swin_stream(x)


class FocalLoss(nn.Module):
    """FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t), with optional label smoothing and mixup support."""

    def __init__(self, alpha=None, gamma=2.0, label_smoothing=0.0, reduction="mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.label_smoothing = label_smoothing
        self.reduction = reduction

    def forward(self, logits, targets):
        C = logits.shape[1]
        if targets.dim() == 1:
            smooth = self.label_smoothing / C
            t = torch.full_like(logits, smooth)
            t.scatter_(1, targets.unsqueeze(1), 1.0 - self.label_smoothing + smooth) if self.label_smoothing > 0 \
                else t.copy_(F.one_hot(targets, C).float())
        else:
            t = targets

        probs = F.softmax(logits, dim=-1)
        loss = -(t * (1 - probs) ** self.gamma * F.log_softmax(logits, dim=-1))

        if self.alpha is not None:
            loss = loss * self.alpha.to(logits.device).unsqueeze(0)

        loss = loss.sum(dim=-1)
        return loss.mean() if self.reduction == "mean" else loss.sum() if self.reduction == "sum" else loss


def build_model(config):
    return FractureMambaViT(num_classes=config["data"]["num_classes"], config=config)
