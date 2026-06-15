"""
ConvCVAE baseline for IES scenario generation (upgraded from MLP-CVAE).

Architecture:
  Encoder: 1D-CNN (5,24) → Conv→BN→ReLU → Conv→BN→ReLU → Flatten → +c(37) → μ/σ(64)
  Decoder: [z(64)+c(37)] → FC → Reshape(64,24) → Conv→BN→ReLU → Conv→Sigmoid → (5,24)

Condition Scaling: mimics CFG (cond_scale) by interpolating latent mean between
  unconditional and conditional encodings during sampling.

Training: β-VAE (β=0.05), AdamW lr=1e-3, CosineAnnealing, 300 epochs
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ConvCVAE(nn.Module):
    """1D-CNN Conditional VAE for 24h × 5-variable time series."""

    def __init__(self, seq_len=24, channels=5, condition_dim=37, latent_dim=64):
        super().__init__()
        self.seq_len = seq_len
        self.channels = channels
        self.condition_dim = condition_dim
        self.latent_dim = latent_dim

        # --- CNN Encoder: (B, 5, 24) → features ---
        self.enc_conv = nn.Sequential(
            nn.Conv1d(channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32), nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64), nn.ReLU(),
            nn.Flatten(),  # → (B, 64*24=1536)
        )
        enc_out_dim = 64 * seq_len  # 1536

        self.fc_mu = nn.Linear(enc_out_dim + condition_dim, latent_dim)
        self.fc_logvar = nn.Linear(enc_out_dim + condition_dim, latent_dim)

        # --- CNN Decoder: z+c → (5, 24) ---
        self.fc_dec_in = nn.Linear(latent_dim + condition_dim, 64 * seq_len)
        self.dec_conv = nn.Sequential(
            nn.Conv1d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32), nn.ReLU(),
            nn.Conv1d(32, channels, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )

    def encode(self, x, c):
        """x: (B, 5, 24), c: (B, 37) → μ, logvar: (B, latent_dim)"""
        h = self.enc_conv(x)
        h = torch.cat([h, c], dim=1)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, c):
        """z: (B, latent_dim), c: (B, 37) → x_hat: (B, 5, 24)"""
        h = self.fc_dec_in(torch.cat([z, c], dim=1))
        h = h.view(-1, 64, self.seq_len)
        return self.dec_conv(h)

    def forward(self, x, c):
        mu, logvar = self.encode(x, c)
        z = self.reparameterize(mu, logvar)
        x_hat = self.decode(z, c)
        return x_hat, mu, logvar

    def sample_with_cfg(self, cond_t, cond_scale=1.5, device='cpu'):
        """Sample with condition scaling (analogous to CFG in diffusion).

        mu_scaled = mu_uncond + cond_scale * (mu_cond - mu_uncond)
        z ~ N(mu_scaled, 0.5 * I)

        Args:
            cond_t: (B, 37) condition tensor
            cond_scale: float, scale factor (1.0 = normal, >1 = stronger conditioning)
            device: torch device

        Returns:
            x_hat: (B, 5, 24) in [0, 1]
        """
        self.eval()
        B = cond_t.shape[0]
        dummy = torch.zeros(B, self.channels, self.seq_len).to(device)
        uncond = torch.zeros(B, self.condition_dim).to(device)

        with torch.no_grad():
            mu_cond, _ = self.encode(dummy, cond_t)
            mu_uncond, _ = self.encode(dummy, uncond)
            mu_scaled = mu_uncond + cond_scale * (mu_cond - mu_uncond)
            z = mu_scaled + 0.5 * torch.randn(B, self.latent_dim).to(device)
            x_hat = self.decode(z, cond_t)
        return x_hat

    def sample(self, cond_t, cond_scale=2.0, device='cpu'):
        """Sample with or without condition scaling.

        Args:
            cond_t: (B, 37) condition tensor
            cond_scale: if > 1.0, use CFG-like conditioning
            device: torch device

        Returns:
            x_hat: (B, 5, 24) in [0, 1]
        """
        if cond_scale > 1.0:
            return self.sample_with_cfg(cond_t, cond_scale=cond_scale, device=device)

        self.eval()
        B = cond_t.shape[0]
        z = torch.randn(B, self.latent_dim).to(device)
        with torch.no_grad():
            x_hat = self.decode(z, cond_t)
        return x_hat


def vae_loss(x_hat, x, mu, logvar, beta=0.05):
    """β-VAE loss: reconstruction MSE + β * KL divergence.

    Args:
        x_hat: (B, C, T) reconstructed
        x: (B, C, T) original
        mu, logvar: (B, latent_dim)
        beta: KL weight (0.05 = weak regularization, focus on reconstruction)

    Returns:
        total_loss, recon_loss, kl_loss
    """
    recon_loss = F.mse_loss(x_hat, x, reduction='mean')
    kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    total = recon_loss + beta * kl_loss
    return total, recon_loss, kl_loss


if __name__ == '__main__':
    model = ConvCVAE(seq_len=24, channels=5, condition_dim=37, latent_dim=64)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"ConvCVAE params: {n_params:,}")

    x = torch.randn(16, 5, 24)
    c = torch.randn(16, 37)
    x_hat, mu, logvar = model(x, c)
    print(f"x: {x.shape}, x_hat: {x_hat.shape}, μ: {mu.shape}, logvar: {logvar.shape}")
    print(f"Output range: [{x_hat.min():.3f}, {x_hat.max():.3f}]")

    total, recon, kl = vae_loss(x_hat, x, mu, logvar)
    print(f"Loss: total={total:.4f}, recon={recon:.4f}, kl={kl:.4f}")

    # Test CFG sampling
    c_cfg = torch.randn(8, 37)
    samples = model.sample(c_cfg, cond_scale=2.0)
    print(f"Sampled (CFG=2.0): {samples.shape}")

    # Test normal sampling
    samples_std = model.sample(c_cfg, cond_scale=1.0)
    print(f"Sampled (no CFG): {samples_std.shape}")
    print("ConvCVAE test PASSED")
