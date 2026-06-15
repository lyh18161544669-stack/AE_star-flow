"""
GAT (Graph Attention Network) Condition Encoder for Energy Variable Interaction Modeling.

Explicitly models the 5 energy variables (wind, solar, electric, heat, cold) as graph nodes,
using multi-head graph attention to learn dynamic inter-variable coupling relationships.

Reference: MS-CGDM (Zhang et al., Energy 2025) and G-DDPM (Miraki et al., Energy and AI 2025)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

FEATURE_COLS = ['wind', 'solar', 'electric', 'heat', 'cold']


class GATLayer(nn.Module):
    """Single Graph Attention Network layer with multi-head attention.

    For each node i, computes attention scores with all neighbors j:
        e_ij = LeakyReLU(a^T [W h_i || W h_j])
        α_ij = softmax_j(e_ij)
        h'_i = σ(mean_over_heads(Σ α_ij W h_j))
    """

    def __init__(self, in_dim, out_dim, num_heads=4, dropout=0.1, concat_heads=True):
        super().__init__()
        self.num_heads = num_heads
        self.out_dim = out_dim
        self.concat_heads = concat_heads

        self.W = nn.Linear(in_dim, out_dim * num_heads, bias=False)
        self.a = nn.Parameter(torch.zeros(2 * out_dim, 1))
        nn.init.xavier_uniform_(self.a)

        self.leaky_relu = nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout(dropout)

        if not concat_heads:
            self.out_proj = nn.Linear(out_dim * num_heads, out_dim, bias=False)

    def forward(self, x, edge_index=None):
        """
        Args:
            x: (batch, num_nodes, in_dim) node features
            edge_index: optional (2, num_edges), if None use fully connected graph
        Returns:
            out: (batch, num_nodes, out_dim * num_heads) if concat_heads
                 (batch, num_nodes, out_dim) if not
        """
        b, n, _ = x.shape

        # Linear transformation: (b, n, out_dim * num_heads)
        Wh = self.W(x).view(b, n, self.num_heads, self.out_dim)

        # Compute attention scores for all pairs (fully connected graph)
        # Wh_i: (b, n, 1, heads, out_dim), Wh_j: (b, 1, n, heads, out_dim)
        Wh_i = Wh.unsqueeze(2)  # (b, n, 1, heads, out_dim)
        Wh_j = Wh.unsqueeze(1)  # (b, 1, n, heads, out_dim)

        # Concatenate pairs: (b, n, n, heads, 2*out_dim)
        pairs = torch.cat([Wh_i.expand(-1, -1, n, -1, -1),
                           Wh_j.expand(-1, n, -1, -1, -1)], dim=-1)

        # Attention scores: (b, n, n, heads, 1)
        e = self.leaky_relu(torch.matmul(pairs, self.a.unsqueeze(0).unsqueeze(0)))
        e = e.squeeze(-1)  # (b, n, n, heads)

        # Softmax over neighbors
        alpha = F.softmax(e, dim=2)  # (b, n, n, heads)
        alpha = self.dropout(alpha)

        # Weighted aggregation: (b, n, heads, out_dim)
        h_new = torch.einsum('bijh,bjhd->bihd', alpha, Wh)

        if self.concat_heads:
            # Concatenate heads: (b, n, out_dim * num_heads)
            h_new = h_new.reshape(b, n, self.out_dim * self.num_heads)
        else:
            # Average heads: (b, n, out_dim)
            h_new = self.out_proj(h_new.reshape(b, n, self.out_dim * self.num_heads))

        return h_new


class GATConditionEncoder(nn.Module):
    """GAT-based condition encoder that models variable interactions explicitly.

    Architecture (following MS-CGDM):
      1. Per-variable feature projection (Linear → GELU)
      2. 2-3 GAT layers with residual connections + LayerNorm
      3. Graph readout (mean pooling over nodes → graph embedding)
      4. Output projection to condition embedding

    Input:  per-variable features (batch, 5, per_var_feat_dim)
    Output: graph condition embedding (batch, output_dim)
    """

    def __init__(self,
                 per_var_feat_dim=5,
                 hidden_dim=64,
                 num_heads=4,
                 num_layers=2,
                 output_dim=64,
                 dropout=0.1):
        super().__init__()
        self.per_var_feat_dim = per_var_feat_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.output_dim = output_dim

        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(per_var_feat_dim, hidden_dim),
            nn.GELU()
        )

        # GAT layers with residual connections
        self.gat_layers = nn.ModuleList()
        self.layer_norms = nn.ModuleList()
        for i in range(num_layers):
            concat = (i < num_layers - 1)  # concat heads in all but last layer
            in_dim = hidden_dim * num_heads if i > 0 else hidden_dim
            out_dim = hidden_dim  # each head outputs hidden_dim
            self.gat_layers.append(
                GATLayer(in_dim, out_dim, num_heads, dropout, concat_heads=concat)
            )
            if concat:
                self.layer_norms.append(nn.LayerNorm(out_dim * num_heads))
            else:
                self.layer_norms.append(nn.LayerNorm(out_dim))

        # Graph readout: pool node embeddings into graph-level embedding
        self.graph_readout_fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, output_dim)
        )

        # Residual projection if input dim != GAT output dim
        self.res_proj = None
        if per_var_feat_dim != hidden_dim:
            self.res_proj = nn.Linear(per_var_feat_dim, hidden_dim)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, node_features):
        """
        Args:
            node_features: (batch, 5, per_var_feat_dim) - per-variable features
                [wind_feats, solar_feats, electric_feats, heat_feats, cold_feats]
        Returns:
            graph_embed: (batch, output_dim) - graph-level condition embedding
        """
        b = node_features.shape[0]

        # Input projection
        h = self.input_proj(node_features)  # (b, 5, hidden_dim)

        # GAT layers with residual
        for i, (gat, ln) in enumerate(zip(self.gat_layers, self.layer_norms)):
            h_new = gat(h)  # (b, 5, hidden_dim * num_heads) or (b, 5, hidden_dim)
            h_new = ln(h_new)
            # Project residual if dims don't match
            if h.shape[-1] != h_new.shape[-1]:
                h_res = self.res_proj(node_features) if self.res_proj is not None else h
                if h_res.shape[-1] != h_new.shape[-1]:
                    h_res = F.linear(h_res,
                                     torch.eye(h_new.shape[-1], h_res.shape[-1],
                                               device=h.device))
                h = h_new + F.gelu(h_res[:, :h_new.shape[1], :h_new.shape[-1]])
            else:
                h = h_new + F.gelu(h)
            h = F.dropout(h, p=0.1, training=self.training)

        # Graph readout: mean pooling over 5 variable nodes + FC
        h_pooled = h.mean(dim=1)  # (b, hidden_dim)
        graph_embed = self.graph_readout_fc(h_pooled)  # (b, output_dim)

        return graph_embed


class DailyVariableFeatureExtractor:
    """Extract per-variable daily features for GAT input.

    For each day and each of the 5 variables, extracts:
      - mean, std, max (3 stats)
      - EVT exceedance flag (1)
      - EVT tail probability (1)
    Total: 5 features per variable x 5 variables = 25 features arranged as (5, 5)
    """

    def __init__(self, feature_extractor_v3):
        """
        Args:
            feature_extractor_v3: DailyFeatureExtractorV3 instance
        """
        self.fe = feature_extractor_v3
        self.n_days = feature_extractor_v3.n_days
        self._cache = None

    def _build_cache(self):
        """Pre-compute per-variable features for all days."""
        all_features = np.zeros((self.n_days, 5, 5), dtype=np.float32)  # (day, var, feat)
        for d in range(self.n_days):
            # Daily stats: 19 dims, layout: [w_mean, w_std, w_max, s_mean, s_std, s_max, ...]
            stats = self.fe.daily_stats_norm[d]

            for v in range(5):
                # mean, std, max from daily stats (normalized)
                mean_val = stats[0 + v * 3]
                std_val = stats[1 + v * 3]
                max_val = stats[2 + v * 3]

                # EVT features
                exceed = self.fe.evt_features[d, v]       # exceedance indicator
                tail_p = self.fe.evt_features[d, 5 + v]    # tail probability

                all_features[d, v, 0] = mean_val
                all_features[d, v, 1] = std_val
                all_features[d, v, 2] = max_val
                all_features[d, v, 3] = exceed
                all_features[d, v, 4] = tail_p

        self._cache = all_features

    def get_node_features(self, day_idx):
        """Get per-variable node features for a given day.

        Returns:
            node_features: np.ndarray (5, 5) - (wind, solar, electric, heat, cold) x feat_dim
        """
        if self._cache is None:
            self._build_cache()
        return self._cache[day_idx].copy()

    def get_node_features_batch(self, day_indices):
        """Get per-variable node features for multiple days.

        Returns:
            node_features: np.ndarray (batch, 5, 5)
        """
        if self._cache is None:
            self._build_cache()
        return self._cache[day_indices].copy()


if __name__ == '__main__':
    # Quick test
    print("Testing GATConditionEncoder...")
    encoder = GATConditionEncoder(
        per_var_feat_dim=5,
        hidden_dim=64,
        num_heads=4,
        num_layers=2,
        output_dim=64
    )
    print(f"  Parameters: {sum(p.numel() for p in encoder.parameters()):,}")

    # Simulate batch of 4 days
    x = torch.randn(4, 5, 5)  # (batch, 5 vars, 5 feats)
    out = encoder(x)
    print(f"  Input shape:  {x.shape}")
    print(f"  Output shape: {out.shape}")
    print(f"  Output range: [{out.min().item():.3f}, {out.max().item():.3f}]")

    # Test with real data if available
    import os
    data_path = "./风光电热冷1年数据.csv"
    if os.path.exists(data_path):
        from dataset_energy_v3 import DailyFeatureExtractorV3
        fe = DailyFeatureExtractorV3(data_path)
        var_extractor = DailyVariableFeatureExtractor(fe)
        node_feats = var_extractor.get_node_features(180)
        print(f"\n  Day 180 node features shape: {node_feats.shape}")
        print(f"  Day 180 per-variable features:")
        for v, name in enumerate(FEATURE_COLS):
            print(f"    {name}: mean={node_feats[v,0]:.3f}, std={node_feats[v,1]:.3f}, "
                  f"max={node_feats[v,2]:.3f}, exceed={node_feats[v,3]:.0f}, "
                  f"tail={node_feats[v,4]:.3f}")
