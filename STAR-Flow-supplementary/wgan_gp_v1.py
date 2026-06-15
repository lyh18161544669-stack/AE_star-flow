"""
V1: Conditional WGAN-GP for IES Energy Scenario Generation.

Architecture:
  Generator G(z, c): z(100) + c(37) → MLP → (5, 24)  [-1, 1]
  Critic D(x, c):    x(5×24=120) + c(37) → MLP → scalar (Wasserstein)

Design decisions:
  - Conditional via concatenation of condition vector (CGAN-style)
  - 37-dim EVT condition (no GAT) — matches V2 baseline
  - Tanh output for [-1, 1] range — compatible with diffusion normalization
  - LayerNorm in Critic (not BatchNorm) — WGAN-GP requires per-sample normalization

Reference: WGAN-GP (Gulrajani et al., NeurIPS 2017)
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch import autograd
import os
import numpy as np


class Generator(nn.Module):
    """Conditional Generator: z(100) + c(37) → (5, 24) in [-1, 1]."""

    def __init__(self, latent_dim=100, condition_dim=37, output_channels=5,
                 output_len=24, hidden_dims=(256, 512)):
        super().__init__()
        self.latent_dim = latent_dim
        self.condition_dim = condition_dim
        self.output_channels = output_channels
        self.output_len = output_len
        self.output_flat = output_channels * output_len  # 120

        input_dim = latent_dim + condition_dim  # 137

        layers = []
        prev_dim = input_dim
        for hd in hidden_dims:
            layers.append(nn.Linear(prev_dim, hd))
            layers.append(nn.BatchNorm1d(hd))
            layers.append(nn.ReLU(inplace=True))
            prev_dim = hd
        layers.append(nn.Linear(prev_dim, self.output_flat))
        self.main = nn.Sequential(*layers)
        self.output_act = nn.Tanh()

    def forward(self, z, c):
        x = torch.cat([z, c], dim=1)
        x = self.main(x)
        x = self.output_act(x)
        return x.view(-1, self.output_channels, self.output_len)


class Critic(nn.Module):
    """Conditional Critic (Discriminator): x(5,24) + c(37) → scalar.

    Uses LayerNorm (not BatchNorm) — WGAN-GP gradient penalty is per-sample.
    """

    def __init__(self, input_channels=5, input_len=24, condition_dim=37,
                 hidden_dims=(256, 128)):
        super().__init__()
        self.input_flat = input_channels * input_len  # 120
        input_dim = self.input_flat + condition_dim  # 157

        layers = []
        prev_dim = input_dim
        for hd in hidden_dims:
            layers.append(nn.Linear(prev_dim, hd))
            layers.append(nn.LayerNorm(hd))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            prev_dim = hd
        layers.append(nn.Linear(prev_dim, 1))  # scalar output, no activation
        self.main = nn.Sequential(*layers)

    def forward(self, x, c):
        x_flat = x.view(x.size(0), -1)
        xc = torch.cat([x_flat, c], dim=1)
        return self.main(xc)


class WGAN_GP:
    """Conditional WGAN with Gradient Penalty.

    Training protocol:
      - n_critic=5 discriminator updates per generator update
      - lambda_gp=10 gradient penalty coefficient
      - Adam optimizer, lr=1e-4, betas=(0.5, 0.9)
    """

    def __init__(self, latent_dim=100, condition_dim=37, channels=5, seq_len=24,
                 lr=1e-4, beta1=0.5, beta2=0.9, n_critic=5, lambda_gp=10,
                 device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.latent_dim = latent_dim
        self.condition_dim = condition_dim
        self.channels = channels
        self.seq_len = seq_len
        self.n_critic = n_critic
        self.lambda_gp = lambda_gp

        self.G = Generator(latent_dim, condition_dim, channels, seq_len).to(self.device)
        self.D = Critic(channels, seq_len, condition_dim).to(self.device)

        self.g_optimizer = optim.Adam(self.G.parameters(), lr=lr,
                                       betas=(beta1, beta2))
        self.d_optimizer = optim.Adam(self.D.parameters(), lr=lr,
                                       betas=(beta1, beta2))

        self.g_iters = 0

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def _gradient_penalty(self, real, fake, c):
        """Compute WGAN-GP gradient penalty on interpolated samples."""
        batch_size = real.size(0)
        eta = torch.rand(batch_size, 1, 1, device=self.device)
        eta = eta.expand_as(real)

        interpolated = eta * real + (1 - eta) * fake
        interpolated.requires_grad_(True)

        prob = self.D(interpolated, c)

        gradients = autograd.grad(
            outputs=prob,
            inputs=interpolated,
            grad_outputs=torch.ones_like(prob),
            create_graph=True,
            retain_graph=True,
        )[0]

        gradients = gradients.view(batch_size, -1)
        grad_norm = gradients.norm(2, dim=1)
        return ((grad_norm - 1) ** 2).mean() * self.lambda_gp

    def train_step(self, real_data, condition):
        """Single training step: update D n_critic times, then update G.

        Args:
            real_data: (B, 5, 24) in [-1, 1]
            condition: (B, 37) EVT condition vector

        Returns:
            dict with losses for logging
        """
        real = real_data.to(self.device)
        c = condition.to(self.device)
        B = real.size(0)
        one = torch.tensor(1.0, device=self.device)
        mone = -one

        # ==================== Train Critic ====================
        d_losses = []
        for _ in range(self.n_critic):
            self.d_optimizer.zero_grad()

            # Real
            d_real = self.D(real, c).mean()
            d_real.backward(mone)

            # Fake
            z = torch.randn(B, self.latent_dim, device=self.device)
            with torch.no_grad():
                fake = self.G(z, c)
            d_fake = self.D(fake, c).mean()
            d_fake.backward(one)

            # Gradient penalty
            gp = self._gradient_penalty(real, fake, c)
            gp.backward()

            self.d_optimizer.step()
            d_losses.append((d_fake - d_real + gp).item())

        d_loss = np.mean(d_losses)
        wasserstein = (d_real - d_fake).item()

        # ==================== Train Generator ====================
        self.g_optimizer.zero_grad()

        z = torch.randn(B, self.latent_dim, device=self.device)
        fake = self.G(z, c)
        g_loss = -self.D(fake, c).mean()  # maximize D(fake)
        g_loss.backward()
        self.g_optimizer.step()

        self.g_iters += 1

        return {
            'd_loss': d_loss,
            'g_loss': g_loss.item(),
            'wasserstein': wasserstein,
        }

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def generate(self, condition, batch_size=None):
        """Generate scenarios conditioned on condition vector.

        Args:
            condition: (B, 37) or (37,) EVT condition vector
            batch_size: int (ignored if condition has batch dim)

        Returns:
            Tensor (B, 5, 24) in [-1, 1]
        """
        self.G.eval()
        if condition.dim() == 1:
            condition = condition.unsqueeze(0)
        B = condition.size(0)
        z = torch.randn(B, self.latent_dim, device=self.device)
        c = condition.to(self.device)
        out = self.G(z, c)
        # Unnormalize from [-1, 1] to [0, 1]
        out_01 = (out + 1.0) * 0.5
        self.G.train()
        return out_01

    # ------------------------------------------------------------------
    # Checkpoint
    # ------------------------------------------------------------------

    def save(self, path):
        torch.save({
            'G': self.G.state_dict(),
            'D': self.D.state_dict(),
            'g_optimizer': self.g_optimizer.state_dict(),
            'd_optimizer': self.d_optimizer.state_dict(),
            'g_iters': self.g_iters,
        }, path)

    def load(self, path):
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.G.load_state_dict(ckpt['G'])
        self.D.load_state_dict(ckpt['D'])
        self.g_optimizer.load_state_dict(ckpt['g_optimizer'])
        self.d_optimizer.load_state_dict(ckpt['d_optimizer'])
        self.g_iters = ckpt.get('g_iters', 0)
        return self


# ==============================================================================
# Self-test
# ==============================================================================
if __name__ == '__main__':
    print("WGAN-GP V1 Self-Test")
    print("=" * 50)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    wgan = WGAN_GP(latent_dim=100, condition_dim=37, channels=5, seq_len=24,
                   device=device)

    # Parameter count
    g_params = sum(p.numel() for p in wgan.G.parameters())
    d_params = sum(p.numel() for p in wgan.D.parameters())
    print(f"Generator: {g_params:,} params")
    print(f"Critic:    {d_params:,} params")
    print(f"Total:     {g_params + d_params:,} params")

    # Test forward pass
    B = 16
    real = torch.randn(B, 5, 24, device=wgan.device)
    cond = torch.randn(B, 37, device=wgan.device)
    z = torch.randn(B, 100, device=wgan.device)

    fake = wgan.G(z, cond)
    print(f"Generator output shape: {fake.shape} (expected: [{B}, 5, 24])")
    print(f"Generator output range: [{fake.min():.3f}, {fake.max():.3f}]")

    d_out = wgan.D(real, cond)
    print(f"Critic output shape: {d_out.shape} (expected: [{B}, 1])")

    # Test train step
    losses = wgan.train_step(real, cond)
    print(f"Train step: d_loss={losses['d_loss']:.4f}, "
          f"g_loss={losses['g_loss']:.4f}, W={losses['wasserstein']:.4f}")

    # Test generation
    gen = wgan.generate(cond[:4])
    print(f"Generate output shape: {gen.shape}, range: [{gen.min():.3f}, {gen.max():.3f}]")

    print("\nAll tests passed!")
