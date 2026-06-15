"""
Continuous-time Gaussian Diffusion adapted for 1D sequence data.
Based on https://openreview.net/attachment?id=2LdBqxc1Yv&name=supplementary_material
Uses log SNR parameterization for improved training stability.
"""
import math
import numpy as np
import torch
from torch import sqrt
from torch import nn
import torch.nn.functional as F
from torch.amp import autocast
from torch.special import expm1

from tqdm import tqdm
from einops import rearrange, repeat, reduce

# helpers

def exists(val):
    return val is not None

def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d

# normalization functions

def normalize_to_neg_one_to_one(img):
    return img * 2 - 1

def unnormalize_to_zero_to_one(t):
    return (t + 1) * 0.5

# diffusion helpers

def right_pad_dims_to(x, t):
    padding_dims = x.ndim - t.ndim
    if padding_dims <= 0:
        return t
    return t.view(*t.shape, *((1,) * padding_dims))

# log(snr) schedules

def log(t, eps=1e-20):
    return torch.log(t.clamp(min=eps))

def beta_linear_log_snr(t):
    return -log(expm1(1e-4 + 10 * (t ** 2)))

def alpha_cosine_log_snr(t, s=0.008):
    return -log((torch.cos((t + s) / (1 + s) * math.pi * 0.5) ** -2) - 1, eps=1e-5)

class ContinuousTimeGaussianDiffusion1D(nn.Module):
    def __init__(
        self,
        model,
        *,
        seq_length,
        channels=None,
        noise_schedule='linear',
        num_sample_steps=500,
        clip_sample_denoised=True,
        min_snr_loss_weight=False,
        min_snr_gamma=5,
        use_evt_loss_weight=False,
        evt_loss_lambda=2.0,
        use_dynamic_evt_weight=False,
        dynamic_evt_lambda=3.0,
        use_physics_loss=False,       # V5: PIDM physics-informed loss
        physics_lambda=0.1,            # V5: physics loss weight
        physics_c=0.1,                 # V5: PIDM scale factor c (higher = stronger physics)
        physics_constraint_fn=None,    # V5: constraint module (PhysicsConstraintSet)
    ):
        super().__init__()
        assert model.random_or_learned_sinusoidal_cond, \
            'model must have random_or_learned_sinusoidal_cond=True for continuous-time diffusion'
        assert not model.self_condition, \
            'self-conditioning not supported yet with continuous-time diffusion'

        self.model = model
        self.channels = channels if channels is not None else model.channels
        self.seq_length = seq_length

        # continuous noise schedule
        if noise_schedule == 'linear':
            self.log_snr = beta_linear_log_snr
        elif noise_schedule == 'cosine':
            self.log_snr = alpha_cosine_log_snr
        else:
            raise ValueError(f'unknown noise schedule {noise_schedule}')

        # sampling
        self.num_sample_steps = num_sample_steps
        self.clip_sample_denoised = clip_sample_denoised

        # min SNR loss weighting (https://arxiv.org/abs/2303.09556)
        self.min_snr_loss_weight = min_snr_loss_weight
        self.min_snr_gamma = min_snr_gamma

        # EVT tail-weighted loss (V2: 静态权重)
        self.use_evt_loss_weight = use_evt_loss_weight
        self.evt_loss_lambda = evt_loss_lambda
        self.register_buffer('tail_weight_matrix', None, persistent=False)

        # EVT动态分位数权重 (V3新增: 基于样本实际值的极端程度)
        self.use_dynamic_evt_weight = use_dynamic_evt_weight
        self.dynamic_evt_lambda = dynamic_evt_lambda
        self._position_quantiles = None  # 由训练脚本注册

        # V5: PIDM physics-informed loss
        self.use_physics_loss = use_physics_loss
        self.physics_lambda = physics_lambda
        self.physics_c = physics_c
        self.physics_constraint_fn = physics_constraint_fn

    @property
    def device(self):
        return next(self.model.parameters()).device

    def set_tail_weight_matrix(self, weight_matrix):
        """设置尾部权重矩阵 (channels, seq_len) 用于EVT加权损失 (V2)"""
        if not isinstance(weight_matrix, torch.Tensor):
            weight_matrix = torch.tensor(weight_matrix, dtype=torch.float32)
        self.tail_weight_matrix = weight_matrix.to(self.device)

    def register_position_quantiles(self, position_quantiles):
        """注册每个(channel, hour)位置的经验分位数数据, 用于V3动态EVT权重

        Args:
            position_quantiles: dict {(c, h): np.array of sorted normalized values}
        """
        self._position_quantiles = position_quantiles

    def _compute_dynamic_tail_weight(self, x_start):
        """根据样本实际值在边际分布中的分位数计算V3动态尾部权重

        Args:
            x_start: (b, c, n) 归一化到[0,1]的原始序列

        Returns:
            weight: (b, c, n) 每个元素的动态尾部权重
        """
        if self._position_quantiles is None:
            return torch.ones_like(x_start)

        device = x_start.device
        x_np = x_start.detach().cpu().numpy()
        weight = np.ones_like(x_np, dtype=np.float32)
        alpha = self.dynamic_evt_lambda

        for c in range(self.channels):
            for h in range(self.seq_length):
                key = (c, h)
                if key not in self._position_quantiles:
                    continue
                sorted_vals = self._position_quantiles[key]
                n = len(sorted_vals)
                for b in range(x_np.shape[0]):
                    val = x_np[b, c, h]
                    rank = np.searchsorted(sorted_vals, val) / n
                    tailness = abs(rank - 0.5) * 2
                    weight[b, c, h] = 1.0 + alpha * tailness

        return torch.FloatTensor(weight).to(device)

    def p_mean_variance(self, x, time, time_next, class_labels=None, cond_scale=1.0):
        log_snr = self.log_snr(time)
        log_snr_next = self.log_snr(time_next)
        c = -expm1(log_snr - log_snr_next)

        squared_alpha, squared_alpha_next = log_snr.sigmoid(), log_snr_next.sigmoid()
        squared_sigma, squared_sigma_next = (-log_snr).sigmoid(), (-log_snr_next).sigmoid()

        alpha, sigma, alpha_next = map(sqrt, (squared_alpha, squared_sigma, squared_alpha_next))

        batch_log_snr = repeat(log_snr, ' -> b', b=x.shape[0])

        # CFG: use classifier-free guidance if enabled
        if cond_scale != 1.0 and hasattr(self.model, 'forward_with_cond_scale'):
            pred_noise, _ = self.model.forward_with_cond_scale(
                x, batch_log_snr, class_labels=class_labels, cond_scale=cond_scale
            )
        else:
            pred_noise = self.model(x, batch_log_snr, class_labels=class_labels)

        if self.clip_sample_denoised:
            x_start = (x - sigma * pred_noise) / alpha
            x_start.clamp_(-1., 1.)
            model_mean = alpha_next * (x * (1 - c) / alpha + c * x_start)
        else:
            model_mean = alpha_next / alpha * (x - c * sigma * pred_noise)

        posterior_variance = squared_sigma_next * c

        return model_mean, posterior_variance

    @torch.no_grad()
    def p_sample(self, x, time, time_next, class_labels=None, cond_scale=1.0):
        batch, *_, device = *x.shape, x.device
        model_mean, model_variance = self.p_mean_variance(
            x=x, time=time, time_next=time_next, class_labels=class_labels,
            cond_scale=cond_scale
        )

        if time_next == 0:
            return model_mean

        noise = torch.randn_like(x)
        return model_mean + sqrt(model_variance) * noise

    @torch.no_grad()
    def p_sample_loop(self, shape, class_labels=None, cond_scale=1.0):
        batch = shape[0]

        img = torch.randn(shape, device=self.device)
        steps = torch.linspace(1., 0., self.num_sample_steps + 1, device=self.device)

        for i in tqdm(range(self.num_sample_steps), desc='sampling loop time step', total=self.num_sample_steps):
            times = steps[i]
            times_next = steps[i + 1]
            img = self.p_sample(img, times, times_next, class_labels=class_labels,
                               cond_scale=cond_scale)

        img.clamp_(-1., 1.)
        img = unnormalize_to_zero_to_one(img)
        return img

    @torch.no_grad()
    def sample(self, batch_size=16, class_labels=None, cond_scale=1.0):
        shape = (batch_size, self.channels, self.seq_length)
        return self.p_sample_loop(shape, class_labels=class_labels,
                                  cond_scale=cond_scale)

    @autocast('cuda', enabled=False)
    def q_sample(self, x_start, times, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))

        log_snr = self.log_snr(times)
        log_snr_padded = right_pad_dims_to(x_start, log_snr)
        alpha, sigma = sqrt(log_snr_padded.sigmoid()), sqrt((-log_snr_padded).sigmoid())
        x_noised = x_start * alpha + noise * sigma

        return x_noised, log_snr

    def random_times(self, batch_size):
        return torch.zeros((batch_size,), device=self.device).float().uniform_(0, 1)

    def p_losses(self, x_start, times, class_labels=None, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))

        x, log_snr = self.q_sample(x_start=x_start, times=times, noise=noise)
        model_out = self.model(x, log_snr, class_labels=class_labels)

        losses = F.mse_loss(model_out, noise, reduction='none')  # (b, c, n)

        # Fix #6: 组合EVT权重使用几何平均, 避免独立相乘导致极端位置权重过大
        # 旧代码: losses *= w_static * w_dynamic → 最大 7×2.5 = 17.5x
        # 新代码: losses *= sqrt(w_static * w_dynamic), clamp max 5.0
        combined_w = None
        if self.use_evt_loss_weight and self.tail_weight_matrix is not None:
            w = self.tail_weight_matrix.to(losses.device)
            combined_w = 1.0 + self.evt_loss_lambda * (w - 1.0)  # [1, 7]

        if self.use_dynamic_evt_weight and self._position_quantiles is not None:
            x_01 = (x_start + 1.0) * 0.5
            dyn_w = self._compute_dynamic_tail_weight(x_01)
            dyn_w = dyn_w.to(losses.device)  # [1, 2.5]
            if combined_w is not None:
                combined_w = torch.sqrt(combined_w.unsqueeze(0) * dyn_w)  # geometric mean
            else:
                combined_w = dyn_w

        if combined_w is not None:
            combined_w = combined_w.clamp(max=5.0)  # hard cap
            losses = losses * combined_w

        losses = reduce(losses, 'b ... -> b', 'mean')

        if self.min_snr_loss_weight:
            snr = log_snr.exp()
            loss_weight = snr.clamp(max=self.min_snr_gamma) / snr
            losses = losses * loss_weight

        # V5: PIDM physics-informed loss on x0 prediction
        if self.use_physics_loss and self.physics_constraint_fn is not None:
            # Estimate clean x0 from noise prediction
            # x = sqrt_alpha * x0 + sigma * noise
            # model_out ~ noise prediction
            # => x0_pred = (x - sigma * model_out) / sqrt_alpha
            log_snr_padded = right_pad_dims_to(x, log_snr)
            sqrt_alpha = log_snr_padded.sigmoid().sqrt()
            sigma = (-log_snr_padded).sigmoid().sqrt()
            pred_x0 = (x - sigma * model_out) / sqrt_alpha.clamp(min=1e-5)
            pred_x0 = pred_x0.clamp(-1., 1.)

            # Convert from [-1, 1] to [0, 1] for physics constraints
            pred_x0_01 = (pred_x0 + 1.0) * 0.5

            # Compute physics residual
            phys_penalty, _phys_components = self.physics_constraint_fn(pred_x0_01)

            # Time-dependent weighting following PIDM Eq.14:
            #   phys_weight = 1 / (2 * Σ̄t) = c / (2 * Σt)
            # In continuous time, Σt ≈ σ² = 1 - ᾱt (noise variance proxy).
            # Clamped to [0.05, 50] for training stability.
            sigma_mean = sigma.mean()
            sigma_sq = (sigma_mean ** 2).clamp(min=1e-4)
            phys_weight = self.physics_c / (2.0 * sigma_sq)
            phys_weight = phys_weight.clamp(0.05, 50.0)

            losses = losses + self.physics_lambda * phys_weight * phys_penalty

        return losses.mean()

    def forward(self, x, class_labels=None, *args, **kwargs):
        b, c, n, device = *x.shape, x.device
        assert n == self.seq_length, f'seq length must be {self.seq_length}, got {n}'

        times = self.random_times(b)
        x = normalize_to_neg_one_to_one(x)
        return self.p_losses(x, times, class_labels=class_labels, *args, **kwargs)
