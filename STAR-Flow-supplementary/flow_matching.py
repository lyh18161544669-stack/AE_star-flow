"""
Continuous-Time Flow Matching for IES Energy Scenario Generation.

Replaces DDPM's noise prediction with velocity field prediction:
  - DDPM: model learns ε̂(x_t, t, c) — "what is the noise?"
  - FM:   model learns v̂(x_t, t, c) — "which way to the data?"

Key advantages:
  1. Faster sampling: 50 ODE steps vs 500 SDE steps (10x)
  2. Deterministic: same input → same output (ODE integration)
  3. More intuitive CFG: velocity-space interpolation
  4. Simpler training: no noise schedule, no SNR weighting

Reference: Lipman et al. (2023), "Flow Matching for Generative Modeling"
            Tong et al. (2023), "Conditional Flow Matching"

Noise schedule: OT-VP (Optimal Transport, x_t = (1-t)*x_0 + t*x_1)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm.auto import tqdm


def unnormalize_to_zero_to_one(x):
    """[-1, 1] → [0, 1]"""
    return (x + 1.0) * 0.5


class ContinuousTimeFlowMatching1D(nn.Module):
    """Flow Matching wrapper for continuous-time generative modeling.

    Compatible with DualChannelDenoiser (or any model with signature:
        forward(x, times, class_labels=None, cond_drop_prob=None)).

    Training:
        diff = ContinuousTimeFlowMatching1D(model, seq_length=24, channels=5)
        loss = diff(x_data, class_labels=cond)  # calls forward()

    Sampling:
        samples = diff.sample(batch_size=100, class_labels=cond, cond_scale=2.0)
    """

    def __init__(
        self,
        model,
        *,
        seq_length,
        channels=None,
        use_evt_loss_weight=False,
        evt_loss_lambda=1.0,
        use_dynamic_evt_weight=False,
        dynamic_evt_lambda=1.5,
    ):
        super().__init__()
        assert hasattr(model, 'random_or_learned_sinusoidal_cond'), \
            'model must have random_or_learned_sinusoidal_cond=True'

        self.model = model
        self.channels = channels if channels is not None else model.channels
        self.seq_length = seq_length

        # EVT tail-weighted loss (fix #6: geometric-mean combination)
        self.use_evt_loss_weight = use_evt_loss_weight
        self.evt_loss_lambda = evt_loss_lambda
        self.register_buffer('tail_weight_matrix', None, persistent=False)

        # EVT dynamic quantile weighting
        self.use_dynamic_evt_weight = use_dynamic_evt_weight
        self.dynamic_evt_lambda = dynamic_evt_lambda
        self._position_quantiles = None

    # ---- EVT weight helpers (shared with DDPM) ----

    def set_tail_weight_matrix(self, weight_matrix):
        """Set pre-computed static tail weights (C, T)."""
        device = next(self.parameters()).device
        self.tail_weight_matrix = weight_matrix.to(device)

    def register_position_quantiles(self, position_quantiles):
        """Register sorted normalized values per (channel, hour)."""
        self._position_quantiles = position_quantiles

    def _compute_dynamic_tail_weight(self, x_data):
        """Dynamic tail weight based on value quantile (fix #3: GPU-native)."""
        if self._position_quantiles is None:
            return torch.ones_like(x_data)

        device = x_data.device
        weight = torch.ones_like(x_data)
        alpha = self.dynamic_evt_lambda

        for c in range(self.channels):
            for h in range(self.seq_length):
                key = (c, h)
                if key not in self._position_quantiles:
                    continue
                sorted_vals = torch.from_numpy(self._position_quantiles[key]).float().to(device)
                n = len(sorted_vals)
                rank = torch.searchsorted(sorted_vals, x_data[:, c, h]) / n
                tailness = (rank - 0.5).abs() * 2
                weight[:, c, h] = 1.0 + alpha * tailness

        return weight

    # ---- Training ----

    def forward(self, x_data, class_labels=None):
        """Flow Matching training loss.

        Args:
            x_data: (b, c, n) in [0, 1] (MinMaxScaler from Dataset)
            class_labels: (b, cond_dim) condition vectors

        Returns:
            scalar MSE loss
        """
        b, c, n = x_data.shape

        # Fix #1: normalize [0,1] → [-1,1] to match DDPM convention and FM OT path
        x_data = x_data * 2 - 1

        # 1. Sample time t ~ U(0, 1)
        t = torch.rand(b, device=x_data.device)

        # 2. Sample noise from standard Gaussian
        x_noise = torch.randn_like(x_data)

        # 3. OT interpolant: x_t = (1-t)*x_noise + t*x_data
        t_padded = t.view(b, 1, 1)
        x_t = (1.0 - t_padded) * x_noise + t_padded * x_data

        # 4. Target velocity field: v = x_data - x_noise (constant along OT path)
        target_v = x_data - x_noise

        # 5. Model prediction: adjust t based on time embedding type
        if hasattr(self.model, 'time_embed_type') and self.model.time_embed_type == 'fm':
            t_model = t  # FMTimeEmbedding accepts [0,1] natively
        else:
            t_model = t * 1000.0  # SinusoidalTimeEmbedding needs scaled values
        pred_v = self.model(x_t, t_model, class_labels=class_labels)

        # 6. Base MSE loss
        losses = F.mse_loss(pred_v, target_v, reduction='none')  # (b, c, n)

        # 7. EVT weighting (fix #6: geometric mean when both active)
        combined_w = None
        if self.use_evt_loss_weight and self.tail_weight_matrix is not None:
            w = self.tail_weight_matrix.to(losses.device)
            combined_w = 1.0 + self.evt_loss_lambda * (w - 1.0)  # [1, 7]

        if self.use_dynamic_evt_weight and self._position_quantiles is not None:
            x_01 = (x_data + 1.0) * 0.5  # [-1,1] → [0,1]
            dyn_w = self._compute_dynamic_tail_weight(x_01)
            if combined_w is not None:
                combined_w = torch.sqrt(combined_w.unsqueeze(0) * dyn_w)
            else:
                combined_w = dyn_w

        if combined_w is not None:
            combined_w = combined_w.clamp(max=5.0)
            losses = losses * combined_w

        losses = losses.mean(dim=(1, 2))  # (b,)
        return losses.mean()

    # ---- Sampling ----

    @torch.no_grad()
    def sample(self, batch_size, class_labels=None, cond_scale=1.0,
               num_steps=50, method='euler', show_progress=True):
        """Generate scenarios by solving the probability flow ODE.

        Args:
            batch_size: number of scenarios to generate
            class_labels: (batch_size, cond_dim) condition vectors
            cond_scale: CFG guidance scale (1.0 = no CFG)
            num_steps: ODE integration steps (50-100 recommended)
            method: 'euler' or 'heun'
            show_progress: show tqdm progress bar

        Returns:
            (batch_size, channels, seq_length) denormalized to [0, 1]
        """
        shape = (batch_size, self.channels, self.seq_length)
        device = next(self.parameters()).device

        # Start from pure Gaussian noise at t=0
        x = torch.randn(shape, device=device)
        dt = 1.0 / num_steps

        iterator = range(num_steps)
        if show_progress:
            iterator = tqdm(iterator, desc='FM sampling', total=num_steps)

        for step in iterator:
            t = step / num_steps + 1e-6  # small epsilon to avoid t=0 boundary
            t_model_val = t if hasattr(self.model, 'time_embed_type') and self.model.time_embed_type == 'fm' else t * 1000.0
            t_tensor = torch.full((batch_size,), t_model_val, device=device)

            # Predict velocity with optional CFG
            if cond_scale != 1.0 and hasattr(self.model, 'forward_with_cond_scale'):
                pred_v, _ = self.model.forward_with_cond_scale(
                    x, t_tensor, class_labels=class_labels, cond_scale=cond_scale
                )
            else:
                pred_v = self.model(x, t_tensor, class_labels=class_labels)

            # Euler step: x_{t+dt} = x_t + dt * v(x_t, t)
            if method == 'euler':
                x = x + dt * pred_v
            elif method == 'heun':
                # Fix #2: Proper Heun's (explicit trapezoidal) method
                # Predictor: Euler step → evaluate at t+dt
                x_pred = x + dt * pred_v
                t_next_val = (t + dt) if hasattr(self.model, 'time_embed_type') and self.model.time_embed_type == 'fm' else (t + dt) * 1000.0
                t_next = torch.full((batch_size,), t_next_val, device=device)
                if cond_scale != 1.0 and hasattr(self.model, 'forward_with_cond_scale'):
                    v_pred, _ = self.model.forward_with_cond_scale(
                        x_pred, t_next, class_labels=class_labels, cond_scale=cond_scale
                    )
                else:
                    v_pred = self.model(x_pred, t_next, class_labels=class_labels)
                # Corrector: average of v(t) and v(t+dt)
                x = x + 0.5 * dt * (pred_v + v_pred)
            else:
                raise ValueError(f"Unknown method: {method}")

        # Clamp and denormalize: [-1, 1] → [0, 1]
        x = x.clamp(-1.0, 1.0)
        x = unnormalize_to_zero_to_one(x)
        return x


# ==============================================================================
# Self-test
# ==============================================================================
if __name__ == '__main__':
    print("=" * 60)
    print("Testing ContinuousTimeFlowMatching1D")
    print("=" * 60)

    from dual_channel_denoiser import DualChannelDenoiser

    model = DualChannelDenoiser(
        seq_len=24, channels=5, condition_dim=37,
        hidden_dim=256, num_heads=4, num_encoder_layers=2,
        num_decoder_layers=4, mlp_ratio=4, dropout=0.1,
        cross_var_module='star', use_hour_embedding=True,
        fusion_type='cross_attn_gate',
    )

    fm = ContinuousTimeFlowMatching1D(
        model,
        seq_length=24,
        channels=5,
        use_evt_loss_weight=True,
        evt_loss_lambda=1.0,
        use_dynamic_evt_weight=False,
    )

    n_params = sum(p.numel() for p in fm.parameters())
    print(f"Total params: {n_params:,}")

    # Test forward
    b = 4
    x = torch.randn(b, 5, 24)
    cond = torch.randn(b, 37)

    model.eval()
    with torch.no_grad():
        loss = fm(x, class_labels=cond)
    print(f"Training loss: {loss.item():.4f}")

    # Test sampling
    with torch.no_grad():
        samples = fm.sample(batch_size=2, class_labels=cond[:2],
                           num_steps=20, show_progress=False)
    print(f"Sample shape: {samples.shape}")
    print(f"Sample range: [{samples.min():.3f}, {samples.max():.3f}]")

    # Test CFG
    with torch.no_grad():
        samples_cfg = fm.sample(batch_size=2, class_labels=cond[:2],
                               cond_scale=2.0, num_steps=20, show_progress=False)
    print(f"CFG sample shape: {samples_cfg.shape}")

    # Test gradient flow
    model.train()
    loss2 = fm(x, class_labels=cond)
    loss2.backward()
    total_grad = sum(p.grad.abs().sum().item() for p in fm.parameters()
                    if p.grad is not None and p.grad.abs().sum() > 0)
    print(f"Gradient norm: {total_grad:.2f}")

    print("\nAll tests passed!")
