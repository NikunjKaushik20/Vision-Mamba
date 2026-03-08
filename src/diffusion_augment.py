
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import os
from tqdm import tqdm


class SinusoidalPosEmb(nn.Module):
    """Sinusoidal positional embeddings for timesteps."""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
    
    def forward(self, t):
        device = t.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = t[:, None] * emb[None, :]
        return torch.cat([emb.sin(), emb.cos()], dim=-1)


class ConvBlock(nn.Module):
    """Convolutional block with GroupNorm and SiLU activation."""
    def __init__(self, in_ch, out_ch, time_emb_dim=None, num_groups=8):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.norm1 = nn.GroupNorm(min(num_groups, out_ch), out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.norm2 = nn.GroupNorm(min(num_groups, out_ch), out_ch)
        self.act = nn.SiLU()
        
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, out_ch),
        ) if time_emb_dim else None
        
        self.residual = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
    
    def forward(self, x, t_emb=None):
        h = self.act(self.norm1(self.conv1(x)))
        if self.time_mlp and t_emb is not None:
            h = h + self.time_mlp(t_emb)[:, :, None, None]
        h = self.act(self.norm2(self.conv2(h)))
        return h + self.residual(x)


class SmallUNet(nn.Module):
    """
    Small U-Net for the diffusion model.
    Class-conditioned via class embeddings added to time embeddings.
    """
    def __init__(self, in_channels=3, base_channels=64, num_classes=2, time_dim=128):
        super().__init__()
        self.time_dim = time_dim
        
        # Time and class embeddings
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(time_dim),
            nn.Linear(time_dim, time_dim * 2),
            nn.SiLU(),
            nn.Linear(time_dim * 2, time_dim),
        )
        self.class_emb = nn.Embedding(num_classes, time_dim)
        
        # Encoder
        self.enc1 = ConvBlock(in_channels, base_channels, time_dim)
        self.enc2 = ConvBlock(base_channels, base_channels * 2, time_dim)
        self.enc3 = ConvBlock(base_channels * 2, base_channels * 4, time_dim)
        
        self.pool = nn.MaxPool2d(2)
        
        # Bottleneck
        self.bottleneck = ConvBlock(base_channels * 4, base_channels * 4, time_dim)
        
        # Decoder
        self.up3 = nn.ConvTranspose2d(base_channels * 4, base_channels * 4, 2, stride=2)
        self.dec3 = ConvBlock(base_channels * 8, base_channels * 2, time_dim)
        self.up2 = nn.ConvTranspose2d(base_channels * 2, base_channels * 2, 2, stride=2)
        self.dec2 = ConvBlock(base_channels * 4, base_channels, time_dim)
        self.up1 = nn.ConvTranspose2d(base_channels, base_channels, 2, stride=2)
        self.dec1 = ConvBlock(base_channels * 2, base_channels, time_dim)
        
        self.out = nn.Conv2d(base_channels, in_channels, 1)
    
    def forward(self, x, t, class_label):
        # Embeddings
        t_emb = self.time_mlp(t) + self.class_emb(class_label)
        
        # Encoder
        e1 = self.enc1(x, t_emb)
        e2 = self.enc2(self.pool(e1), t_emb)
        e3 = self.enc3(self.pool(e2), t_emb)
        
        # Bottleneck
        b = self.bottleneck(self.pool(e3), t_emb)
        
        # Decoder
        d3 = self.up3(b)
        # Handle size mismatches
        if d3.shape != e3.shape:
            d3 = F.interpolate(d3, size=e3.shape[2:])
        d3 = self.dec3(torch.cat([d3, e3], dim=1), t_emb)
        
        d2 = self.up2(d3)
        if d2.shape != e2.shape:
            d2 = F.interpolate(d2, size=e2.shape[2:])
        d2 = self.dec2(torch.cat([d2, e2], dim=1), t_emb)
        
        d1 = self.up1(d2)
        if d1.shape != e1.shape:
            d1 = F.interpolate(d1, size=e1.shape[2:])
        d1 = self.dec1(torch.cat([d1, e1], dim=1), t_emb)
        
        return self.out(d1)


class DiffusionAugmentor:
    """Conditional DDPM for generating synthetic X-ray samples per class."""

    def __init__(self, img_size=64, num_classes=2, timesteps=500, device="cuda"):
        self.img_size = img_size
        self.num_classes = num_classes
        self.timesteps = timesteps
        self.device = device
        
        # Small UNet for the diffusion model
        self.model = SmallUNet(
            in_channels=3, base_channels=64,
            num_classes=num_classes, time_dim=128
        ).to(device)
        
        # Noise schedule
        self.betas = torch.linspace(1e-4, 0.02, timesteps).to(device)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
    
    def add_noise(self, x, t):
        sqrt_alpha = self.sqrt_alphas_cumprod[t][:, None, None, None]
        sqrt_one_minus = self.sqrt_one_minus_alphas_cumprod[t][:, None, None, None]
        noise = torch.randn_like(x)
        return sqrt_alpha * x + sqrt_one_minus * noise, noise
    
    def train_diffusion(self, dataloader, epochs=50, lr=1e-4, save_path=None):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.model.train()
        
        print(f"\n[DIFFUSION] Training diffusion augmentor ({epochs} epochs, img_size={self.img_size})...")
        
        for epoch in range(epochs):
            total_loss = 0
            count = 0
            
            for images, labels in dataloader:
                # Resize to diffusion model size
                images = F.interpolate(images, size=self.img_size, mode="bilinear", align_corners=False)
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                # Random timesteps
                t = torch.randint(0, self.timesteps, (images.shape[0],), device=self.device)
                
                # Add noise
                noisy_images, noise = self.add_noise(images, t)
                
                # Predict noise
                predicted_noise = self.model(noisy_images, t.float(), labels)
                
                loss = F.mse_loss(predicted_noise, noise)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                count += 1
            
            if (epoch + 1) % 10 == 0:
                avg_loss = total_loss / max(count, 1)
                print(f"  Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}")
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save(self.model.state_dict(), save_path)
            print(f"  [SAVE] Diffusion model saved to {save_path}")
    
    @torch.no_grad()
    def generate(self, num_samples=16, class_id=0, output_size=224):
        self.model.eval()
        
        # Start from noise
        x = torch.randn(num_samples, 3, self.img_size, self.img_size).to(self.device)
        labels = torch.full((num_samples,), class_id, dtype=torch.long, device=self.device)
        
        # Reverse diffusion
        for t in reversed(range(self.timesteps)):
            t_batch = torch.full((num_samples,), t, dtype=torch.float32, device=self.device)
            
            predicted_noise = self.model(x, t_batch, labels)
            
            alpha = self.alphas[t]
            alpha_cumprod = self.alphas_cumprod[t]
            beta = self.betas[t]
            
            if t > 0:
                noise = torch.randn_like(x)
            else:
                noise = torch.zeros_like(x)
            
            x = (1 / torch.sqrt(alpha)) * (
                x - (beta / torch.sqrt(1 - alpha_cumprod)) * predicted_noise
            ) + torch.sqrt(beta) * noise
        
        # Upsample to output size
        if self.img_size != output_size:
            x = F.interpolate(x, size=output_size, mode="bilinear", align_corners=False)
        
        # Clamp to valid range
        x = torch.clamp(x, -2.5, 2.5)  # Roughly within normalized range
        
        return x, labels
