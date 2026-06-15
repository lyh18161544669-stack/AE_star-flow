"""
V4 双通道去噪网络: 时序Transformer + 特征Transformer + STAR融合
参考: STDR-DiT / AugDiT (Xie et al., Energy 2026)

架构:
  Input (b, 5, 24) + time + condition(101)
    → Temporal channel: Self-attention over 24 time steps
    → Feature channel: Self-attention over 5 variables + STAR module
    → DiT decoder blocks with AdaLN-Zero modulation
    → Additive fusion → Output (b, 5, 24)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat


# ==============================================================================
# Building Blocks
# ==============================================================================

class SinusoidalTimeEmbedding(nn.Module):
    """Sinusoidal timestep embedding + 2-layer MLP, follows DiT/DDPM convention.

    Designed for log-SNR values (range ~[-5, 10]) with max_period=10000.
    """

    def __init__(self, dim, max_period=10000):
        super().__init__()
        self.dim = dim
        self.max_period = max_period
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.SiLU(),
            nn.Linear(dim * 4, dim),
        )

    def forward(self, t):
        """
        Args:
            t: (b,) log-SNR values (continuous time)
        Returns:
            emb: (b, dim)
        """
        half = self.dim // 2
        freqs = torch.exp(-math.log(self.max_period) *
                          torch.arange(0, half, dtype=torch.float32, device=t.device) / half)
        args = t.float().unsqueeze(-1) * freqs.unsqueeze(0)  # (b, half)
        emb = torch.cat([args.sin(), args.cos()], dim=-1)    # (b, dim)
        return self.mlp(emb)


class FMTimeEmbedding(nn.Module):
    """Time embedding designed for Flow Matching with t ∈ [0, 1].

    Uses frequency range [exp(0)=1, exp(-4.6)≈0.01] appropriate for [0,1] inputs.
    For t∈[0,1]:
      - freq=1.0 creates ~0.16 oscillation cycles (smooth variation)
      - freq=0.01 creates nearly flat response (captures slow drift)

    No t-scaling needed — designed natively for [0,1] range.
    """

    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.SiLU(),
            nn.Linear(dim * 4, dim),
        )

    def forward(self, t):
        """
        Args:
            t: (b,) values in [0, 1]
        Returns:
            emb: (b, dim)
        """
        half = self.dim // 2
        # Frequencies: exp(-linspace(0, 4.6, half)) = [1.0, ~0.01]
        freqs = torch.exp(
            -torch.linspace(0, 4.6, half, dtype=torch.float32, device=t.device)
        )
        args = t.float().unsqueeze(-1) * freqs.unsqueeze(0)  # (b, half)
        emb = torch.cat([args.sin(), args.cos()], dim=-1)    # (b, dim)
        return self.mlp(emb)


class AdaLNZero(nn.Module):
    """Adaptive Layer Normalization with Zero-initialized modulation.

    Given a context vector, produces scale/shift/gate for attention and MLP sublayers.
    Zero-initialization ensures identity mapping at start of training.
    """

    def __init__(self, dim, context_dim):
        super().__init__()
        self.proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(context_dim, 6 * dim)
        )
        # Zero-init the final layer for identity at initialization
        nn.init.zeros_(self.proj[-1].weight)
        nn.init.zeros_(self.proj[-1].bias)

    def forward(self, context):
        """
        Args:
            context: (b, context_dim)
        Returns:
            Tuple of 6 tensors, each (b, 1, dim)
        """
        params = self.proj(context)  # (b, 6*dim)
        gamma1, beta1, alpha1, gamma2, beta2, alpha2 = params.chunk(6, dim=-1)
        return (
            gamma1.unsqueeze(1), beta1.unsqueeze(1), alpha1.unsqueeze(1),
            gamma2.unsqueeze(1), beta2.unsqueeze(1), alpha2.unsqueeze(1),
        )


class DiTBlock(nn.Module):
    """Diffusion Transformer Block with AdaLN-Zero modulation."""

    def __init__(self, dim, num_heads=4, mlp_ratio=4, context_dim=256, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            dim, num_heads, dropout=dropout, batch_first=True
        )
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * mlp_ratio),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * mlp_ratio, dim),
            nn.Dropout(dropout),
        )
        self.adaln = AdaLNZero(dim, context_dim)

    def forward(self, x, context):
        """
        Args:
            x: (b, n, dim) input sequence
            context: (b, context_dim) condition + time embedding
        Returns:
            (b, n, dim) transformed sequence
        """
        gamma1, beta1, alpha1, gamma2, beta2, alpha2 = self.adaln(context)

        # Self-attention with AdaLN
        h_norm = self.norm1(x)
        h_scaled = h_norm * (1 + gamma1) + beta1
        attn_out, _ = self.attn(h_scaled, h_scaled, h_scaled)
        x = x + alpha1 * attn_out

        # MLP with AdaLN
        h_norm = self.norm2(x)
        h_scaled = h_norm * (1 + gamma2) + beta2
        mlp_out = self.mlp(h_scaled)
        x = x + alpha2 * mlp_out

        return x


class TransformerEncoder(nn.Module):
    """Standard Transformer Encoder: multi-head self-attention + FFN + residual + LN."""

    def __init__(self, dim, num_heads=4, mlp_ratio=4, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            dim, num_heads, dropout=dropout, batch_first=True
        )
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * mlp_ratio),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * mlp_ratio, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        h = self.norm1(x)
        attn_out, _ = self.attn(h, h, h)
        x = x + attn_out
        h = self.norm2(x)
        x = x + self.mlp(h)
        return x


class STARModule(nn.Module):
    """STar Aggregate Redistribute module for cross-variable information sharing.

    Centralized aggregation–redistribution mechanism:
    1. Project each variable's features to latent space
    2. Pool via softmax-weighted sum (deterministic) or stochastic sampling
    3. Broadcast core representation to all variables
    4. Concatenate with original and transform
    """

    def __init__(self, dim, dropout=0.1):
        super().__init__()
        self.proj_in = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
        )
        self.proj_out = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
        )

    def forward(self, x):
        """
        Args:
            x: (b, n_vars, dim) feature channel representation
        Returns:
            (b, n_vars, dim) enriched representation
        """
        b, n, d = x.shape

        # Project to latent
        h = self.proj_in(x)  # (b, n, d)

        # Softmax importance weights over variables
        weights = F.softmax(h.mean(dim=-1, keepdim=True), dim=1)  # (b, n, 1)

        # Weighted aggregation: core representation
        core = (h * weights).sum(dim=1, keepdim=True)  # (b, 1, d)

        # Broadcast and concatenate
        core_broadcast = core.expand(-1, n, -1)  # (b, n, d)
        fused = torch.cat([x, core_broadcast], dim=-1)  # (b, n, 2d)

        return self.proj_out(fused)


class IESGraphAttention(nn.Module):
    """IES物理拓扑约束的图注意力, 替代STAR的全局softmax汇总。

    与V4的GATConditionEncoder的区分:
      - V4 GAT: 从5维统计特征学习全连接图 → 拼到条件向量 (模型之前)
      - IES-GAT: 从256维学习表征沿物理边传递消息 → 精炼特征 (模型之中)

    信息流方向 (查询方向, 非能量流方向):
      wind ──→ electric ←── solar     wind/solar → electric (电网查询风光)
                │ ↕                          ↕        (CHP双向信息交换)
                ↓                            ↓
              heat ──→ cold            electric/heat → cold (冷负荷查询冷源)

    attn_mask[i, j] = 0 表示"节点i可以查询节点j获取信息":
      - electric 查询 wind/solar: 电网调度需要知道风光出力
      - cold 查询 electric/heat: 冷负荷需要知道冷源状态
      - wind/solar 仅查询自身: 独立可再生能源, 不需要跨变量信息
    """

    # 信息查询邻接矩阵 (5×5 bool): True = 节点i可以查询节点j
    # 节点: wind(0), solar(1), electric(2), heat(3), cold(4)
    # adj[i, j]=True → i queries j for information (i receives from j)
    PHYSICAL_ADJ = torch.tensor([
        # w  s  e  h  c
        [ 1, 0, 0, 0, 0],  # wind: self only (独立源)
        [ 0, 1, 0, 0, 0],  # solar: self only (独立源)
        [ 1, 1, 1, 1, 1],  # electric: queries all (IES枢纽, 包含EC→Cold)
        [ 0, 0, 1, 1, 0],  # heat: ←electric,self
        [ 0, 0, 1, 1, 1],  # cold: ←electric,heat,self
    ], dtype=torch.bool)

    def __init__(self, dim, num_heads=4, dropout=0.1):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads

        # Fix #3: register adjacency as persistent buffer
        self.register_buffer('physical_adj', self.PHYSICAL_ADJ.clone())

        # Multi-head attention with physical topology mask
        self.attn = nn.MultiheadAttention(
            dim, num_heads, dropout=dropout, batch_first=True
        )

        # Layer norms
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

        # FFN after message passing
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 2, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        """
        Args:
            x: (b, 5, dim) 5个变量的特征表征
        Returns:
            (b, 5, dim) 物理拓扑消息传递后的表征
        """
        b, n, d = x.shape
        h = self.norm1(x)

        # Fix #2: fp16-safe mask. Must match input dtype for MHA compat
        # -1e4 is representable in fp16 (max=65504), exp(-1e4)≈0 in softmax
        attn_mask = torch.zeros(n, n, device=x.device, dtype=x.dtype)
        attn_mask[~self.physical_adj] = -1e4

        # Multi-head attention constrained by physical topology
        attn_out, _ = self.attn(h, h, h, attn_mask=attn_mask)

        # Residual + FFN (Transformer-style)
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x


class CrossAttentionFusion(nn.Module):
    """Cross-attention + gated fusion for temporal-feature channel integration.

    Replaces simple additive fusion with:
      1. Cross-attention: variable tokens (5) query time tokens (24)
         -> time-aware variable representations
      2. Cross-attention: time tokens (24) query variable tokens (5)
         -> variable-aware temporal representations
      3. Learnable gate: per (var, time) position decides weighting
         between the two expanded representations

    This captures variable x time cross-dependencies that simple addition cannot:
      - Solar at noon vs solar at midnight -> different temporal context
      - Wind interacting with electric load -> time-specific coupling patterns
    """

    def __init__(self, dim, num_heads=4, dropout=0.1):
        super().__init__()
        # Variable (5 tokens) queries time (24 tokens)
        self.cross_var2time = nn.MultiheadAttention(
            dim, num_heads, dropout=dropout, batch_first=True
        )
        # Time (24 tokens) queries variable (5 tokens)
        self.cross_time2var = nn.MultiheadAttention(
            dim, num_heads, dropout=dropout, batch_first=True
        )

        # Layer norms for residual connections
        self.norm_feat = nn.LayerNorm(dim)
        self.norm_time = nn.LayerNorm(dim)

        # Gate network: per-position fusion weight
        self.gate_proj = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, 1),
        )

    def forward(self, h_time, h_feat, return_attention=False):
        """
        Args:
            h_time: (b, 24, dim) temporal channel output (after decoder)
            h_feat: (b, 5, dim)  feature channel output (after decoder)
            return_attention: if True, also return attention weight dict
        Returns:
            fused: (b, 5, 24, dim) cross-attended + gated representation
            attn_dict (optional): dict with 'var2time' and 'time2var' attention weights
        """
        # Parallel cross-attention from ORIGINAL representations (fix #4)
        # Both directions computed simultaneously — no information asymmetry
        h_feat_cross, attn_v2t = self.cross_var2time(h_feat, h_time, h_time)
        h_time_cross, attn_t2v = self.cross_time2var(h_time, h_feat, h_feat)

        # Independent residual updates
        h_feat = self.norm_feat(h_feat + h_feat_cross)  # (b, 5, dim)
        h_time = self.norm_time(h_time + h_time_cross)  # (b, 24, dim)

        # Expand to (b, 5, 24, dim) for per-position fusion
        h_time_exp = h_time.unsqueeze(1).expand(-1, 5, -1, -1)   # (b, 5, 24, dim)
        h_feat_exp = h_feat.unsqueeze(2).expand(-1, -1, 24, -1)  # (b, 5, 24, dim)

        # Gated fusion: per-position sigmoid gate
        concat = torch.cat([h_feat_exp, h_time_exp], dim=-1)     # (b, 5, 24, 2*dim)
        gate = torch.sigmoid(self.gate_proj(concat))             # (b, 5, 24, 1)
        fused = gate * h_feat_exp + (1 - gate) * h_time_exp      # (b, 5, 24, dim)

        if return_attention:
            return fused, {'var2time': attn_v2t, 'time2var': attn_t2v}
        return fused


# ==============================================================================
# Main Denoiser
# ==============================================================================

class DualChannelDenoiser(nn.Module):
    """V4 Dual-Channel Diffusion Transformer Denoiser.

    Replaces KarrasUnet1D with a dual-channel architecture:
    - Temporal channel: Transformer encoder + DiT decoder over 24 time steps
    - Feature channel: Transformer encoder + STAR + DiT decoder over 5 variables
    - Additive fusion of both channels
    - Output projection to (5, 24)

    Compatible with ContinuousTimeGaussianDiffusion1D.
    """

    def __init__(self,
                 seq_len=24,
                 channels=5,
                 condition_dim=101,
                 hidden_dim=256,
                 num_heads=4,
                 num_encoder_layers=2,
                 num_decoder_layers=4,
                 mlp_ratio=4,
                 dropout=0.1,
                 cross_var_module='star',  # 'star', 'ies_gat', or 'none'
                 cond_drop_prob=0.0,       # CFG: probability of dropping condition during training
                 use_hour_embedding=False, # P1a: add sin/cos hour channels to input
                 fusion_type='additive',   # 'additive' or 'cross_attn_gate'
                 time_embed_type='sinusoidal'):  # 'sinusoidal' (DDPM) or 'fm' (Flow Matching)
        super().__init__()

        self.seq_len = seq_len
        self.channels = channels
        self.hidden_dim = hidden_dim
        self.condition_dim = condition_dim
        self.cross_var_module = cross_var_module
        self.use_star = (cross_var_module == 'star')  # backward compat
        self.cond_drop_prob = cond_drop_prob
        self.use_hour_embedding = use_hour_embedding
        self.fusion_type = fusion_type
        self.time_embed_type = time_embed_type

        # Required attributes for diffusion model compatibility
        self.random_or_learned_sinusoidal_cond = True
        self.self_condition = False

        # Time embedding (sinusoidal for DDPM, FM-specific for Flow Matching)
        if time_embed_type == 'fm':
            self.time_emb = FMTimeEmbedding(hidden_dim)
        else:
            self.time_emb = SinusoidalTimeEmbedding(hidden_dim)

        # Condition embedding
        self.cond_emb = nn.Sequential(
            nn.Linear(condition_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
        )
        # CFG: learnable null condition for classifier-free guidance
        self.null_condition_emb = nn.Parameter(torch.randn(condition_dim))
        # Combined context: time + condition
        self.context_proj = nn.Linear(hidden_dim * 2, hidden_dim)

        # ---- Input projections ----
        # Temporal: project data channels (+ optional hour embedding) → hidden_dim
        input_channels = channels + (2 if use_hour_embedding else 0)
        self.temporal_proj = nn.Linear(input_channels, hidden_dim)
        self.temporal_pos = nn.Parameter(torch.randn(1, seq_len, hidden_dim) * 0.02)

        # Feature: project 24 time steps → hidden_dim for each variable
        self.feature_proj = nn.Linear(seq_len, hidden_dim)
        self.variable_pos = nn.Parameter(torch.randn(1, channels, hidden_dim) * 0.02)  # fix #5

        # ---- Encoders ----
        self.temporal_encoder = nn.ModuleList([
            TransformerEncoder(hidden_dim, num_heads, mlp_ratio, dropout)
            for _ in range(num_encoder_layers)
        ])
        self.feature_encoder = nn.ModuleList([
            TransformerEncoder(hidden_dim, num_heads, mlp_ratio, dropout)
            for _ in range(num_encoder_layers)
        ])

        # Cross-variable information sharing in feature channel
        if cross_var_module == 'star':
            self.cross_var = STARModule(hidden_dim, dropout)
        elif cross_var_module == 'ies_gat':
            self.cross_var = IESGraphAttention(hidden_dim, num_heads, dropout)
        else:
            self.cross_var = None  # no cross-variable module
        # ---- Decoders (DiT blocks with AdaLN-Zero) ----
        self.temporal_decoder = nn.ModuleList([
            DiTBlock(hidden_dim, num_heads, mlp_ratio, hidden_dim, dropout)
            for _ in range(num_decoder_layers)
        ])
        self.feature_decoder = nn.ModuleList([
            DiTBlock(hidden_dim, num_heads, mlp_ratio, hidden_dim, dropout)
            for _ in range(num_decoder_layers)
        ])

        # ---- Fusion module ----
        if fusion_type == 'cross_attn_gate':
            self.fusion = CrossAttentionFusion(hidden_dim, num_heads, dropout)
        else:
            self.fusion = None  # use simple additive fusion

        # ---- Output projection ----
        # Fuse temporal and feature representations, then project to output
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),  # 1 value per (channel, time) position
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        # Restore AdaLN-Zero zero-initialization (overwritten by xavier_uniform above)
        for m in self.modules():
            if isinstance(m, AdaLNZero):
                nn.init.zeros_(m.proj[-1].weight)
                nn.init.zeros_(m.proj[-1].bias)

    def _build_context(self, times, class_labels, cond_drop_prob=None):
        """Build combined context embedding from time and condition."""
        t_emb = self.time_emb(times)  # (b, hidden_dim)

        cond = class_labels
        if cond is not None:
            # CFG condition dropout
            cdp = cond_drop_prob if cond_drop_prob is not None else self.cond_drop_prob
            if cdp > 0 and self.training:
                batch = cond.shape[0]
                keep_mask = torch.rand(batch, device=cond.device) > cdp
                null_cond = self.null_condition_emb.unsqueeze(0).expand(batch, -1)
                cond = torch.where(keep_mask.unsqueeze(-1), cond, null_cond)
            c_emb = self.cond_emb(cond)  # (b, hidden_dim)
        else:
            c_emb = torch.zeros_like(t_emb)

        context = torch.cat([t_emb, c_emb], dim=-1)  # (b, 2*hidden_dim)
        context = self.context_proj(context)           # (b, hidden_dim)
        return context

    def forward(self, x, times, class_labels=None, cond_drop_prob=None, return_attention=False):
        """
        Args:
            x: (b, channels, seq_len) noisy input in [-1, 1]
            times: (b,) log-SNR values (continuous time)
            class_labels: (b, condition_dim) condition vector
            cond_drop_prob: optional override for CFG condition dropout probability

        Returns:
            (b, channels, seq_len) noise prediction
        """
        b = x.shape[0]

        # Context embedding (with optional CFG dropout)
        context = self._build_context(times, class_labels, cond_drop_prob)  # (b, hidden_dim)

        # ---- Temporal channel ----
        # (b, 5, 24) → (b, 24, 5+2) → (b, 24, hidden_dim)
        if self.use_hour_embedding:
            hours = torch.arange(self.seq_len, device=x.device).float() / self.seq_len
            hour_sin = torch.sin(2 * math.pi * hours)
            hour_cos = torch.cos(2 * math.pi * hours)
            hour_emb = torch.stack([hour_sin, hour_cos], dim=0).unsqueeze(0).expand(b, -1, -1)
            x_t_in = torch.cat([x, hour_emb], dim=1)  # (b, 7, 24)
        else:
            x_t_in = x
        x_t = rearrange(x_t_in, 'b c n -> b n c')  # (b, 24, c)
        h_time = self.temporal_proj(x_t)             # (b, 24, hidden_dim)
        h_time = h_time + self.temporal_pos           # add learned positional encoding

        for enc in self.temporal_encoder:
            h_time = enc(h_time)                # (b, 24, hidden_dim)

        for dec in self.temporal_decoder:
            h_time = dec(h_time, context)       # (b, 24, hidden_dim)

        # ---- Feature channel ----
        # (b, 5, 24) → (b, 5, hidden_dim)
        x_f = x  # (b, 5, 24)
        h_feat = self.feature_proj(x_f) + self.variable_pos  # (b, 5, hidden_dim) + fix #5

        for enc in self.feature_encoder:
            h_feat = enc(h_feat)                # (b, 5, hidden_dim)

        if self.cross_var is not None:
            h_feat = self.cross_var(h_feat)       # (b, 5, hidden_dim)

        for dec in self.feature_decoder:
            h_feat = dec(h_feat, context)       # (b, 5, hidden_dim)

        # ---- Fusion ----
        attn_dict = None
        if self.fusion is not None:
            # Cross-attention + gated fusion (V3+)
            if return_attention:
                h_fused, attn_dict = self.fusion(h_time, h_feat, return_attention=True)
            else:
                h_fused = self.fusion(h_time, h_feat)  # (b, 5, 24, hidden_dim)
        else:
            # Simple additive fusion (V2.5)
            h_time_exp = h_time.unsqueeze(1).expand(-1, self.channels, -1, -1)
            h_feat_exp = h_feat.unsqueeze(2).expand(-1, -1, self.seq_len, -1)
            h_fused = h_time_exp + h_feat_exp       # (b, 5, 24, hidden_dim)

        # ---- Output projection ----
        h_fused = self.output_norm(h_fused)
        out = self.output_proj(h_fused)          # (b, 5, 24, 1)
        out = out.squeeze(-1)                    # (b, 5, 24)

        if return_attention and attn_dict is not None:
            return out, attn_dict
        return out

    def forward_with_cond_scale(
        self,
        x,
        times,
        class_labels,
        cond_scale = 1.0,
        **kwargs
    ):
        """CFG sampling: interpolate between conditional and unconditional predictions."""
        return_attention = kwargs.get('return_attention', False)

        cond_output = self.forward(x, times, class_labels=class_labels,
                                   cond_drop_prob=0.0, **kwargs)
        if return_attention:
            cond_output, cond_attn = cond_output

        if cond_scale == 1.0:
            if return_attention:
                return cond_output, cond_attn
            return cond_output, None

        null_output = self.forward(x, times, class_labels=class_labels,
                                   cond_drop_prob=1.0, **kwargs)
        if return_attention:
            null_output, null_attn = null_output

        scaled_output = null_output + cond_scale * (cond_output - null_output)
        if return_attention:
            return scaled_output, cond_attn  # conditional pathway attention
        return scaled_output, null_output


# ==============================================================================
# Test
# ==============================================================================

if __name__ == '__main__':
    print("=" * 60)
    print("Testing DualChannelDenoiser")
    print("=" * 60)

    model = DualChannelDenoiser(
        seq_len=24,
        channels=5,
        condition_dim=101,
        hidden_dim=256,
        num_heads=4,
        num_encoder_layers=2,
        num_decoder_layers=4,
        mlp_ratio=4,
        dropout=0.1,
        cross_var_module='star',
    )

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {n_params:,}")

    # Test forward pass
    b = 4
    x = torch.randn(b, 5, 24)
    t = torch.rand(b)
    cond = torch.randn(b, 101)

    model.eval()
    with torch.no_grad():
        out = model(x, t, cond)

    print(f"Input shape:  {x.shape}")
    print(f"Output shape: {out.shape}")
    print(f"Output range: [{out.min():.3f}, {out.max():.3f}]")

    # Verify required attributes
    print(f"random_or_learned_sinusoidal_cond: {model.random_or_learned_sinusoidal_cond}")
    print(f"self_condition: {model.self_condition}")
    print(f"channels: {model.channels}")

    # Test gradient flow
    model.train()
    out = model(x, t, cond)
    loss = out.mean()
    loss.backward()

    total_grad = 0.0
    for name, p in model.named_parameters():
        if p.grad is not None:
            total_grad += p.grad.abs().sum().item()
    print(f"Total gradient norm: {total_grad:.2f}")

    # Cross-var module ablation
    for mod_name in ['star', 'ies_gat', 'none']:
        m = DualChannelDenoiser(
            seq_len=24, channels=5, condition_dim=101,
            hidden_dim=256, num_heads=4, num_encoder_layers=2,
            num_decoder_layers=4, mlp_ratio=4, dropout=0.1,
            cross_var_module=mod_name,
        )
        p = sum(pn.numel() for pn in m.parameters())
        cv = type(m.cross_var).__name__ if m.cross_var else 'None'
        print(f"  {mod_name}: {p:,} params ({cv})")

    print("\nAll tests passed!")
