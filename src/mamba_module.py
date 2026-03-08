import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from functools import partial


class PatchEmbedding(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_channels=3, embed_dim=192):
        super().__init__()
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.proj(x)
        x = rearrange(x, "b c h w -> b (h w) c")
        return self.norm(x)


class SelectiveSSM(nn.Module):
    """
    Pure-PyTorch S6 selective scan. No mamba-ssm CUDA dependency.
    Uses chunked parallelism for speed — vectorized within 32-token chunks,
    sequential across chunks.
    """

    def __init__(self, d_model, d_state=16, d_conv=4, expand_factor=2, dropout=0.1):
        super().__init__()
        self.d_state = d_state
        self.d_inner = int(d_model * expand_factor)

        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)
        self.conv1d = nn.Conv1d(
            self.d_inner, self.d_inner,
            kernel_size=d_conv, padding=d_conv - 1,
            groups=self.d_inner, bias=True,
        )

        # B, C, dt projections
        self.x_proj = nn.Linear(self.d_inner, d_state * 2 + 1, bias=False)

        # A: HiPPO-inspired diagonal init
        A = torch.arange(1, d_state + 1, dtype=torch.float32).unsqueeze(0).expand(self.d_inner, -1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.d_inner))

        self.dt_proj = nn.Linear(1, self.d_inner, bias=True)
        nn.init.uniform_(self.dt_proj.weight, -0.02, 0.02)
        dt = torch.exp(torch.rand(self.d_inner) * (math.log(0.1) - math.log(0.001)) + math.log(0.001))
        with torch.no_grad():
            self.dt_proj.bias.copy_(dt + torch.log(-torch.expm1(-dt)))

        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def selective_scan(self, u, delta, A, B, C, D):
        B_sz, L, d_inner = u.shape
        d_state = A.shape[1]
        CHUNK = 32

        delta_A = torch.exp(delta.unsqueeze(-1) * A.unsqueeze(0).unsqueeze(0)).clamp(max=1.0)
        delta_B_u = delta.unsqueeze(-1) * B.unsqueeze(2) * u.unsqueeze(-1)

        ys = []
        h = torch.zeros(B_sz, d_inner, d_state, device=u.device, dtype=u.dtype)

        for start in range(0, L, CHUNK):
            end = min(start + CHUNK, L)
            dA = delta_A[:, start:end]
            dBu = delta_B_u[:, start:end]
            Cc = C[:, start:end]

            log_dA = torch.log(dA.clamp(min=1e-8))
            cum_log_dA = torch.cumsum(log_dA, dim=1)
            cum_dA = torch.exp(cum_log_dA.clamp(max=20.0))

            inv = torch.exp(-cum_log_dA.clamp(-20.0, 20.0))
            states = cum_dA * (h.unsqueeze(1) + torch.cumsum(dBu * inv, dim=1))

            ys.append((states * Cc.unsqueeze(2)).sum(-1))
            h = states[:, -1]

        y = torch.cat(ys, dim=1)
        return y + u * D.unsqueeze(0).unsqueeze(0)

    def forward(self, x):
        B, L, _ = x.shape

        xz = self.in_proj(x)
        x_in, z = xz.chunk(2, dim=-1)

        # causal conv
        xc = rearrange(x_in, "b l d -> b d l")
        xc = self.conv1d(xc)[:, :, :L]
        xc = F.silu(rearrange(xc, "b d l -> b l d"))

        ssm_out = self.x_proj(xc)
        B_p = ssm_out[:, :, :self.d_state]
        C_p = ssm_out[:, :, self.d_state:2 * self.d_state]
        dt = F.softplus(self.dt_proj(ssm_out[:, :, -1:]))
        A = -torch.exp(self.A_log)

        y = self.selective_scan(xc, dt, A, B_p, C_p, self.D)
        y = self.out_proj(self.dropout(y * F.silu(z)))
        return y


class DropPath(nn.Module):
    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if not self.training or self.drop_prob == 0.0:
            return x
        keep = 1 - self.drop_prob
        mask = torch.rand((x.shape[0],) + (1,) * (x.ndim - 1), device=x.device, dtype=x.dtype)
        return x / keep * mask.floor_() + keep


class MambaBlock(nn.Module):
    def __init__(self, d_model, d_state=16, d_conv=4, expand_factor=2, dropout=0.1, drop_path=0.0):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.ssm = SelectiveSSM(d_model, d_state, d_conv, expand_factor, dropout)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x):
        return x + self.drop_path(self.ssm(self.norm(x)))


class BidirectionalMambaBlock(nn.Module):
    """Process sequence in both directions and fuse — helps catch fracture cues at any orientation."""

    def __init__(self, d_model, d_state=16, d_conv=4, expand_factor=2, dropout=0.1, drop_path=0.0):
        super().__init__()
        kw = dict(d_state=d_state, d_conv=d_conv, expand_factor=expand_factor, dropout=dropout, drop_path=drop_path)
        self.fwd = MambaBlock(d_model, **kw)
        self.bwd = MambaBlock(d_model, **kw)
        self.fusion = nn.Linear(d_model * 2, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        fwd_out = self.fwd(x)
        bwd_out = self.bwd(x.flip(1)).flip(1)
        return self.norm(self.fusion(torch.cat([fwd_out, bwd_out], dim=-1)))


class VisionMamba(nn.Module):
    """
    Vim encoder: patches → learnable pos embed → stack of bidirectional Mamba blocks.
    Returns a CLS token + all patch tokens.
    """

    def __init__(
        self,
        img_size=224, patch_size=16, in_channels=3,
        embed_dim=192, depth=12, d_state=16, d_conv=4,
        expand_factor=2, dropout=0.1, drop_path_rate=0.2,
    ):
        super().__init__()
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        n = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, n + 1, embed_dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        self.pos_drop = nn.Dropout(dropout)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList([
            BidirectionalMambaBlock(embed_dim, d_state, d_conv, expand_factor, dropout, dpr[i])
            for i in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)
        x = torch.cat([self.cls_token.expand(B, -1, -1), x], dim=1)
        x = self.pos_drop(x + self.pos_embed)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return x[:, 0], x[:, 1:]
