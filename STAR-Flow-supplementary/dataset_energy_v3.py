"""
Energy Dataset V3: EVT引导的条件构造器 + 动态分位数尾部权重
用于方向A: 极端场景引导的条件扩散模型 (阶段2: EVT引导采样)

V3 相比 V2 的新增功能:
  1. build_extreme_condition(): 构造指定极端程度的合成条件向量
     - extreme_level ∈ [0, 1] 控制全局极端程度 (0=中位, 0.95=P95, 0.99=P99)
     - 支持 per-variable dict 控制: {'wind': 0.99, 'solar': 0.5, ...}
  2. 动态分位数尾部权重: 基于实际样本值在边际分布中的分位数计算loss权重
     - 取代V2的静态 tail_weight_matrix (仅基于位置尾部厚度)
     - 同时保留静态权重作为baseline

条件向量 (37维):
  - 日统计特征 (19维)
  - 日聚类标签 (5维)
  - 季节编码 (2维)
  - EVT尾部特征 (10维): 超越指示(5) + GPD尾部概率(5)
  - 综合极端标识 (1维)

数据划分 (月分层):
  - Test:  每月1日 (12天) — 场景生成的条件来源
  - Val:   每月3/4/5日 (36天) — 验证loss，选best checkpoint
  - Train: 其余日 (317天, 87%) — 训练模型参数
"""

import torch
import numpy as np
import pandas as pd
import os
import pickle
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans

FEATURE_COLS = ['wind', 'solar', 'electric', 'heat', 'cold']


# ==============================================================================
# Shared data loading utilities
# ==============================================================================

def load_energy_data(data_path):
    """Load energy CSV, auto-detecting format (new with time column or old 5-column).

    Returns:
        raw_data: np.ndarray (N, 5) [wind, solar, electric, heat, cold]
        time_index: pd.DatetimeIndex or None (None for old-format CSV)
    """
    data_path = os.path.normpath(data_path)
    df = pd.read_csv(data_path, encoding='gbk')

    # Detect format: new CSV has 'time' column, old CSV has 5 unnamed columns
    if 'time' in df.columns:
        time_index = pd.to_datetime(df['time'])
        data = df[FEATURE_COLS].values.astype(np.float64)
    elif len(df.columns) >= 6 and df.columns[0].strip().lower() in ('time', 'datetime'):
        time_index = pd.to_datetime(df.iloc[:, 0])
        data_cols = [c for c in df.columns if c.lower() in FEATURE_COLS]
        data = df[data_cols].values.astype(np.float64)
    else:
        # Old format: no header, 5 columns
        time_index = None
        df = pd.read_csv(data_path, header=None, encoding='gbk')
        df.columns = FEATURE_COLS
        data = df[FEATURE_COLS].values.astype(np.float64)

    return data, time_index


def get_monthly_stratified_split(data_path):
    """Compute monthly-stratified train/val/test day indices.

    Test:  1st of each month (12 days) — condition source for generation
    Val:   3rd, 4th, 5th of each month (36 days)
    Train: remaining days (317 days, 87%)

    Returns:
        train_days: np.ndarray of day indices (0-based)
        val_days: np.ndarray of day indices
        test_days: np.ndarray of day indices
    """
    _, time_index = load_energy_data(data_path)

    if time_index is None:
        raise ValueError(
            "Monthly stratified split requires a CSV with a 'time' column. "
            "The old 5-column CSV format is not supported for splitting. "
            "Use the new '源荷数据集.csv' instead."
        )

    n_days = len(time_index) // 24
    day_of_month = time_index.dt.day.values

    train_days, val_days, test_days = [], [], []
    for d in range(n_days):
        dom = day_of_month[d * 24]  # day-of-month at first hour of each day
        if dom == 1:
            test_days.append(d)
        elif dom in (3, 4, 5):
            val_days.append(d)
        else:
            train_days.append(d)

    return np.array(train_days), np.array(val_days), np.array(test_days)


def get_test_day_index(data_path, month, day=1):
    """Get the day index for a specific month/day (e.g., month=1 day=1 → Jan 1).

    Returns:
        int: day index (0-based), or None if not found
    """
    _, time_index = load_energy_data(data_path)
    if time_index is None:
        return None
    for d in range(len(time_index) // 24):
        dt = time_index[d * 24]
        if dt.month == month and dt.day == day:
            return d
    return None


class DailyFeatureExtractorV3:
    """V3: 日级统计特征 + EVT尾部特征 + 极端程度可控条件构造器"""

    def __init__(self, data_path, cache_path=None, n_clusters=5, evt_threshold_pct=95):
        data_path = os.path.normpath(data_path)
        raw_data, self.time_index = load_energy_data(data_path)
        self.raw_data = raw_data
        self.n_days = len(self.raw_data) // 24
        self.n_clusters = n_clusters
        self.evt_threshold_pct = evt_threshold_pct
        self._dynamic_weight_available = False
        self._position_quantiles = {}

        self.cache_path = cache_path or data_path.replace('.csv', '_daily_features_v3.pkl')
        if os.path.exists(self.cache_path):
            self._load_cache()
        else:
            self._compute_all()
            self._save_cache()

    def _compute_all(self):
        self.daily_stats_raw = self._extract_daily_stats_raw()
        self.stats_scaler = MinMaxScaler()
        self.daily_stats_norm = self.stats_scaler.fit_transform(self.daily_stats_raw)
        self.cluster_labels = self._cluster_days()
        self.cluster_centers = self.stats_scaler.inverse_transform(
            self.kmeans.cluster_centers_
        )
        self._fit_evt()
        self._compute_tail_weight_matrix()       # V2静态权重(保留)
        self._compute_dynamic_quantile_weights()  # V3新增: 动态分位数权重
        self._fix_daily_means_norm()              # P1c: 日均值对齐到输出空间 (must run before node_features)
        self._compute_node_features_cache()       # V3.1: GAT节点特征缓存

    def _extract_daily_stats_raw(self):
        stats_list = []
        for d in range(self.n_days):
            day = self.raw_data[d * 24:(d + 1) * 24]
            w, s, e, h, c = day[:, 0], day[:, 1], day[:, 2], day[:, 3], day[:, 4]
            features = [
                w.mean(), w.std(), w.max(),
                s.mean(), s.std(), s.max(),
                e.mean(), e.std(), e.max(), e.max() - e.min(),
                h.mean(), h.std(), h.max(),
                c.mean(), c.std(), c.max(),
                np.max(np.abs(np.diff(w))),
                np.argmax(s) / 23.0,
                np.argmax(e) / 23.0,
            ]
            stats_list.append(features)
        return np.array(stats_list, dtype=np.float32)

    def _fix_daily_means_norm(self):
        """P1c: Replace daily means with hourly-normalized versions.

        Root cause: daily_stats_norm uses a per-day MinMaxScaler (maps 365 daily
        means to [0,1]), while the model output is in hourly MinMaxScaler space.
        For solar, this creates a 2.7x mismatch (dim=0.86 must map to output=0.31).

        Fix: overwrite the daily mean entries in daily_stats_norm with values
        normalized by the hourly MinMaxScaler range, putting them in the same
        [0,1] space as the model output.

        Affected indices in daily_stats (19-dim):
          0: wind mean, 3: solar mean, 6: electric mean,
          10: heat mean, 13: cold mean
        """
        # Hourly data ranges (from MinMaxScaler on 8760h)
        hourly_min = np.array([0.0, 0.0, 651.1, 204.8, 140.7], dtype=np.float32)
        hourly_max = np.array([1000.0, 1206.1, 1556.2, 935.8, 1681.3], dtype=np.float32)
        mean_indices = [0, 3, 6, 10, 13]  # indices in daily_stats for each variable's mean

        self.daily_stats_norm_fixed = self.daily_stats_norm.copy()
        for var_i, idx in enumerate(mean_indices):
            raw_means = self.daily_stats_raw[:, idx]  # original kW
            normed = (raw_means - hourly_min[var_i]) / (hourly_max[var_i] - hourly_min[var_i])
            self.daily_stats_norm_fixed[:, idx] = np.clip(normed, 0.0, 1.0)

    def _cluster_days(self):
        self.kmeans = KMeans(n_clusters=self.n_clusters, random_state=42, n_init=10)
        return self.kmeans.fit_predict(self.daily_stats_norm)

    def _fit_evt(self):
        """对每个变量拟合GPD, 提取EVT尾部特征 (同V2)"""
        try:
            from scipy.stats import genpareto
        except ImportError:
            raise ImportError('EVT需要scipy: pip install scipy')

        self.evt_params = {}
        self.evt_features = np.zeros((self.n_days, 10), dtype=np.float32)

        for var_idx in range(5):
            daily_max = self.daily_stats_raw[:, 2 + var_idx * 3]
            threshold = np.percentile(daily_max, self.evt_threshold_pct)
            exceedances = daily_max[daily_max > threshold] - threshold

            if len(exceedances) >= 10:
                xi, loc, sigma = genpareto.fit(exceedances, floc=0)
            else:
                xi, sigma = 0.0, exceedances.std() if len(exceedances) > 0 else 1.0

            self.evt_params[var_idx] = (threshold, float(xi), float(sigma))

            for d in range(self.n_days):
                day_max = daily_max[d]
                if day_max > threshold:
                    self.evt_features[d, var_idx] = 1.0
                    excess = day_max - threshold
                    tail_prob = 1.0 - genpareto.cdf(excess, xi, loc=0, scale=sigma)
                    self.evt_features[d, 5 + var_idx] = np.float32(np.clip(tail_prob, 1e-6, 1.0))
                else:
                    self.evt_features[d, var_idx] = 0.0
                    self.evt_features[d, 5 + var_idx] = 1.0

        self.extreme_labels = (self.evt_features[:, :5].sum(axis=1) > 0).astype(np.float32)

    def _compute_tail_weight_matrix(self):
        """静态尾部权重矩阵 (V2兼容)"""
        alpha = 2.0
        self.tail_weight_matrix = np.ones((5, 24), dtype=np.float32)
        for c in range(5):
            for h in range(24):
                values = self.raw_data[h::24, c]
                p5, p95, p99 = np.percentile(values, 5), np.percentile(values, 95), np.percentile(values, 99)
                tail_ratio = (p99 - p95) / (p95 - p5 + 1e-8)
                self.tail_weight_matrix[c, h] = 1.0 + alpha * np.clip(tail_ratio, 0.0, 3.0)

    # ======================= V3 新增功能 =======================

    def _compute_dynamic_quantile_weights(self):
        """预计算归一化空间每个(channel, hour)位置的经验CDF分位数断点

        用于训练时根据样本实际值计算动态尾部权重:
        - 值越接近尾部(高分位数或低分位数), loss权重越高
        - 这比静态权重更精准, 因为直接针对样本的极端程度加权
        """
        # 需要访问归一化后的数据 — 在EnergyDataset1DV3中调用
        self._position_quantiles = {}  # {(c, h): np.array of sorted values}
        self._dynamic_weight_available = False

    def compute_position_quantiles(self, normalized_data):
        """在EnergyDataset1DV3初始化后调用, 传入归一化后的8760x5数据

        Args:
            normalized_data: np.ndarray (8760, 5), MinMax归一化后的数据
        """
        for c in range(5):
            for h in range(24):
                values = normalized_data[h::24, c]  # 该位置所有值
                self._position_quantiles[(c, h)] = np.sort(values)
        self._dynamic_weight_available = True

    # ======================= V3.1: GAT节点特征 =======================

    def _compute_node_features_cache(self):
        """预计算每个日的逐变量节点特征 (5变量 × 5特征), 供GAT条件编码器使用

        每个变量提取5个特征:
          - mean, std, max (来自日统计归一化值)
          - EVT超越指示 (0/1)
          - EVT尾部概率 (1.0=非极端, 越小越极端)
        """
        self._node_features_cache = np.zeros((self.n_days, 5, 5), dtype=np.float32)
        for d in range(self.n_days):
            stats = self.daily_stats_norm_fixed[d]  # P1c: use hourly-aligned stats
            for v in range(5):
                self._node_features_cache[d, v, 0] = stats[0 + v * 3]  # mean
                self._node_features_cache[d, v, 1] = stats[1 + v * 3]  # std
                self._node_features_cache[d, v, 2] = stats[2 + v * 3]  # max
                self._node_features_cache[d, v, 3] = self.evt_features[d, v]       # exceed
                self._node_features_cache[d, v, 4] = self.evt_features[d, 5 + v]   # tail prob

    def get_node_features(self, day_idx):
        """获取指定日的GAT节点特征

        Returns:
            np.ndarray (5, 5): 5变量 × 5特征的节点特征矩阵
        """
        if not hasattr(self, '_node_features_cache'):
            self._compute_node_features_cache()
        return self._node_features_cache[day_idx].copy()

    def get_node_features_batch(self, day_indices):
        """批量获取GAT节点特征

        Returns:
            np.ndarray (batch, 5, 5)
        """
        if not hasattr(self, '_node_features_cache'):
            self._compute_node_features_cache()
        return self._node_features_cache[day_indices].copy()

    def get_dynamic_tail_weight(self, x_normalized):
        """根据归一化样本值计算动态尾部权重

        Args:
            x_normalized: torch.Tensor (b, c, n), 归一化到[0,1]的序列

        Returns:
            weight: torch.Tensor (b, c, n), 每个元素的尾部权重
        """
        if not self._dynamic_weight_available:
            return torch.ones_like(x_normalized)

        device = x_normalized.device
        x_np = x_normalized.detach().cpu().numpy()
        weight = np.ones_like(x_np, dtype=np.float32)
        alpha = 3.0  # 动态权重放大系数

        for c in range(5):
            for h in range(24):
                sorted_vals = self._position_quantiles[(c, h)]
                n = len(sorted_vals)
                # 用searchsorted计算每个样本值的分位数
                for b in range(x_np.shape[0]):
                    val = x_np[b, c, h]
                    rank = np.searchsorted(sorted_vals, val) / n
                    # 尾部权重: 远离中位数的值获得更高权重
                    tailness = abs(rank - 0.5) * 2  # [0, 1], 0=中位, 1=极端
                    weight[b, c, h] = 1.0 + alpha * tailness

        return torch.FloatTensor(weight).to(device)

    # ======================= EVT引导的条件构造器 =======================

    def build_extreme_condition(self, day_idx=180, extreme_level=0.5,
                                 per_variable=None):
        """构造指定极端程度的合成条件向量 (EVT引导采样的核心)

        通过操纵条件向量中的EVT特征(超越指示 + GPD尾部概率),
        控制生成场景的极端程度, 无需重新训练模型。

        季节性约束 (P1b): 根据模板日的季节自动限制哪些变量的EVT被操纵:
          - 夏季 (5-9月, 制冷季): 仅操纵 cold, electric, solar 的EVT
          - 冬季 (11-3月, 供暖季): 仅操纵 heat, electric 的EVT
          - 过渡季 (4, 10月): 全局操纵所有变量

        Args:
            day_idx: int, 模板日索引(0-364), 提供日统计+聚类+季节基线
            extreme_level: float [0, 1], 全局极端程度
              0.0 = 中位条件 (所有超越指示=0, 尾部概率=1.0)
              0.5 = 中等 (超越指示=0.5, 尾部概率=范围中点)
              0.9 = P90极端 (超越指示=1, 尾部概率=0.1)
              0.95 = P95极端 (超越指示=1, 尾部概率=0.05)
              0.99 = P99极端 (超越指示=1, 尾部概率=0.01)
            per_variable: dict or None, 逐变量极端程度
              {'wind': 0.99, 'solar': 0.5, 'electric': 0.0, 'heat': 0.5, 'cold': 0.0}
              如果提供, 覆盖 extreme_level 的全局设置

        Returns:
            condition: np.ndarray (37,), 合成条件向量

        Examples:
            >>> fe = DailyFeatureExtractorV3(data_path)
            >>> # 全局P95极端
            >>> cond = fe.build_extreme_condition(180, extreme_level=0.95)
            >>> # 仅风电极端, 其他正常
            >>> cond = fe.build_extreme_condition(180, extreme_level=0.0,
            ...         per_variable={'wind': 0.99})
        """
        # 基线条件 (日统计 + 聚类 + 季节)
        daily_stat = self.daily_stats_norm_fixed[day_idx]  # P1c: aligned to hourly space
        cluster_1h = np.zeros(self.n_clusters, dtype=np.float32)
        cluster_1h[self.cluster_labels[day_idx]] = 1.0
        doy = day_idx
        season_feat = np.float32([np.sin(2 * np.pi * doy / 365.0),
                                   np.cos(2 * np.pi * doy / 365.0)])

        # P1b: 季节性EVT操纵 — 确定该日所属季节及活跃变量
        # 使用时间索引推断月份 (day 0 = Jan 1)
        if self.time_index is not None:
            month = self.time_index[day_idx * 24].month
        else:
            month = (day_idx * 24) // (365 * 24 / 12) + 1  # fallback

        if month in (5, 6, 7, 8, 9):
            # 夏季/制冷季: 冷负荷、电负荷、光伏为主导
            season_active_vars = {'cold', 'electric', 'solar'}
        elif month in (11, 12, 1, 2, 3):
            # 冬季/供暖季: 热负荷、电负荷为主导
            season_active_vars = {'heat', 'electric'}
        else:
            # 过渡季 (4, 10月): 全局操纵
            season_active_vars = None

        # 构造合成EVT特征
        # P1b fix: 活跃变量用合成极端值, 非活跃变量保留模板日的真实EVT特征
        # 这样条件向量保持在训练分布内, 避免 OOD (out-of-distribution) 问题
        real_evt = self.evt_features[day_idx]  # 模板日的真实EVT (10,)
        synth_evt = np.zeros(10, dtype=np.float32)

        for var_idx in range(5):
            var_name = FEATURE_COLS[var_idx]

            # 非活跃季节变量: 使用真实EVT (保持训练分布)
            if season_active_vars is not None and var_name not in season_active_vars:
                synth_evt[var_idx] = real_evt[var_idx]
                synth_evt[5 + var_idx] = real_evt[5 + var_idx]
                continue

            level = extreme_level
            if per_variable is not None and var_name in per_variable:
                level = per_variable[var_name]

            level = np.clip(level, 0.0, 0.999)

            # Fix #2: exceedance must be BINARY to match training distribution
            # Training: evt_features[var_idx] ∈ {0, 1}
            # Old code used continuous exceedance (e.g., 0.6) → OOD
            # Threshold lowered from 0.5 to 0.0: any λ>0 triggers EVT manipulation,
            # enabling smooth control across the full [0, 0.95] range (Section 5.3)
            if level > 0.0:
                exceedance = 1.0
                # GPD-based tail probability: inverse mapping from desired quantile
                params = self.evt_params.get(var_idx)
                if params is not None:
                    _, xi, sigma = params
                    # level = P(day_max ≤ q_level) → excess = q_level - threshold
                    # excess = genpareto.ppf(level, xi, loc=0, scale=sigma)
                    # tail_prob = 1 - genpareto.cdf(excess, xi, loc=0, scale=sigma)
                    # Simplified: tail_prob ≃ 1 - level (asymptotically correct for GPD)
                    # But use actual GPD for accuracy when ξ ≠ 0
                    try:
                        excess = genpareto.ppf(level, xi, loc=0, scale=sigma)
                        tail_prob = 1.0 - genpareto.cdf(excess, xi, loc=0, scale=sigma)
                        tail_prob = float(np.clip(tail_prob, 1e-6, 1.0))
                    except Exception:
                        tail_prob = 1.0 - level
                else:
                    tail_prob = 1.0 - level
            else:
                exceedance = 0.0
                tail_prob = 1.0  # non-extreme: no tail

            synth_evt[var_idx] = np.float32(exceedance)
            synth_evt[5 + var_idx] = np.float32(np.clip(tail_prob, 1e-6, 1.0))

        # 综合极端标识
        combined_extreme = 1.0 if extreme_level > 0.5 or \
            (per_variable is not None and any(v > 0.5 for v in per_variable.values())) else 0.0

        condition = np.concatenate([
            daily_stat.astype(np.float32),      # 19
            cluster_1h,                          # 5
            season_feat,                         # 2
            synth_evt.astype(np.float32),        # 10
            np.float32([combined_extreme])       # 1
        ])
        return condition

    def build_extreme_condition_batch(self, batch_size, day_idx=180,
                                       extreme_level=0.5, per_variable=None,
                                       noise_std=0.0):
        """批量构造极端条件向量, 支持加噪声增加多样性

        Args:
            batch_size: int, 批次大小
            day_idx: int or list, 模板日索引
            extreme_level: float, 全局极端程度
            per_variable: dict or None
            noise_std: float, 对日统计特征添加高斯噪声的标准差

        Returns:
            conditions: np.ndarray (batch_size, 37)
        """
        conditions = np.zeros((batch_size, self.condition_dim), dtype=np.float32)
        for i in range(batch_size):
            d = day_idx if isinstance(day_idx, int) else day_idx[i]
            cond = self.build_extreme_condition(d, extreme_level, per_variable)
            if noise_std > 0:
                cond[:19] += np.random.randn(19).astype(np.float32) * noise_std
            conditions[i] = cond
        return conditions

    def get_evt_level_description(self, extreme_level, per_variable=None):
        """返回极端程度的可读描述"""
        desc = f"extreme_level={extreme_level}"
        if per_variable:
            desc += f", per_var={per_variable}"
        desc += "\n各变量EVT特征:"
        for var_idx, name in enumerate(FEATURE_COLS):
            level = extreme_level
            if per_variable and name in per_variable:
                level = per_variable[name]
            thresh, xi, sigma = self.evt_params[var_idx]
            if level > 0.5:
                # 计算对应此极端程度的GPD分位数
                try:
                    from scipy.stats import genpareto
                    target_prob = level
                    gpd_quantile = genpareto.ppf(target_prob, xi, loc=0, scale=sigma)
                    target_value = thresh + gpd_quantile
                    desc += f"\n  {name}: P{int(level*100)}极端, "
                    desc += f"目标日最大值≈{target_value:.1f} (阈值={thresh:.1f})"
                except:
                    desc += f"\n  {name}: P{int(level*100)}极端"
            else:
                desc += f"\n  {name}: 非极端 (P{int(level*100)})"
        return desc

    # ======================= V2兼容方法 =======================

    def get_condition(self, day_idx):
        """构建37维条件向量 (使用真实EVT特征, 同V2)"""
        daily_stat = self.daily_stats_norm_fixed[day_idx]  # P1c: aligned to hourly space
        cluster_1h = np.zeros(self.n_clusters, dtype=np.float32)
        cluster_1h[self.cluster_labels[day_idx]] = 1.0
        doy = day_idx
        season_feat = np.float32([np.sin(2 * np.pi * doy / 365.0),
                                   np.cos(2 * np.pi * doy / 365.0)])
        evt_feat = self.evt_features[day_idx]
        extreme = np.array([self.extreme_labels[day_idx]], dtype=np.float32)
        return np.concatenate([
            daily_stat.astype(np.float32),
            cluster_1h,
            season_feat,
            evt_feat.astype(np.float32),
            extreme
        ])

    @property
    def condition_dim(self):
        return 19 + self.n_clusters + 2 + 10 + 1  # 37

    def get_tail_weight_tensor(self):
        return torch.FloatTensor(self.tail_weight_matrix)

    def _save_cache(self):
        cache = {
            'daily_stats_raw': self.daily_stats_raw,
            'stats_scaler': self.stats_scaler,
            'daily_stats_norm': self.daily_stats_norm,
            'daily_stats_norm_fixed': self.daily_stats_norm_fixed,
            'cluster_labels': self.cluster_labels,
            'cluster_centers': self.cluster_centers,
            'kmeans': self.kmeans,
            'extreme_labels': self.extreme_labels,
            'evt_params': self.evt_params,
            'evt_features': self.evt_features,
            'evt_threshold_pct': self.evt_threshold_pct,
            'tail_weight_matrix': self.tail_weight_matrix,
            'n_clusters': self.n_clusters,
            'n_days': self.n_days,
            'node_features_cache': getattr(self, '_node_features_cache', None),
        }
        with open(self.cache_path, 'wb') as f:
            pickle.dump(cache, f)
        print(f"V3日级特征缓存已保存至: {self.cache_path}")

    def _load_cache(self):
        with open(self.cache_path, 'rb') as f:
            cache = pickle.load(f)
        self.daily_stats_raw = cache['daily_stats_raw']
        self.stats_scaler = cache['stats_scaler']
        self.daily_stats_norm = cache['daily_stats_norm']
        # P1c: fixed daily means (aligned to hourly space)
        if 'daily_stats_norm_fixed' in cache:
            self.daily_stats_norm_fixed = cache['daily_stats_norm_fixed']
        else:
            self._fix_daily_means_norm()
            self._compute_node_features_cache()  # recompute with fixed stats
        self.cluster_labels = cache['cluster_labels']
        self.cluster_centers = cache['cluster_centers']
        self.kmeans = cache['kmeans']
        self.extreme_labels = cache['extreme_labels']
        self.evt_params = cache.get('evt_params', {})
        self.evt_features = cache.get('evt_features',
                                       np.zeros((self.n_days, 10), dtype=np.float32))
        self.evt_threshold_pct = cache.get('evt_threshold_pct', 95)
        self.tail_weight_matrix = cache.get('tail_weight_matrix',
                                             np.ones((5, 24), dtype=np.float32))
        self.n_clusters = cache['n_clusters']
        self._node_features_cache = cache.get('node_features_cache', None)
        print(f"从缓存加载V3日级特征: {self.cache_path}")

    def print_summary(self):
        print(f"\n{'='*60}")
        print(f"V3 日级特征提取器 (EVT引导采样)")
        print(f"{'='*60}")
        print(f"总天数: {self.n_days}")
        print(f"条件向量维度: {self.condition_dim}")
        print(f"EVT阈值百分位: P{self.evt_threshold_pct}")
        print(f"综合极端日数: {self.extreme_labels.sum():.0f}/{self.n_days} "
              f"({self.extreme_labels.mean()*100:.1f}%)")
        print(f"\n各变量GPD参数:")
        for var_idx, name in enumerate(FEATURE_COLS):
            if var_idx in self.evt_params:
                thresh, xi, sigma = self.evt_params[var_idx]
                exceed_count = self.evt_features[:, var_idx].sum()
                print(f"  {name}: 阈值={thresh:.2f}, xi={xi:+.3f}, sigma={sigma:.3f}, "
                      f"超越天数={exceed_count:.0f}")
        print(f"\n静态尾部权重范围: [{self.tail_weight_matrix.min():.2f}, "
              f"{self.tail_weight_matrix.max():.2f}]")
        print(f"动态分位数权重: {'可用' if self._dynamic_weight_available else '待计算'}")
        print(f"{'='*60}\n")


class EnergyDataset1DV3(torch.utils.data.Dataset):
    """数据集V3: 返回 (序列 [5, 24], 条件向量 [37], GAT节点特征 [5, 5])

    与V2兼容, 额外支持:
      - 构造合成极端条件进行EVT引导采样
      - 提供动态分位数尾部权重给loss函数
      - 月分层train/val/test划分
    """

    def __init__(self, data_path, seq_len=24, normalize=True,
                 feature_extractor=None, split=None):
        self.seq_len = seq_len
        self.normalize = normalize

        raw_data, self.time_index = load_energy_data(data_path)
        self.data = raw_data.astype(np.float32)

        if self.normalize:
            self.scaler = MinMaxScaler()
            self.data_normalized = self.scaler.fit_transform(self.data)
        else:
            self.data_normalized = self.data.astype(np.float32)
            self.scaler = None

        if feature_extractor is not None:
            self.fe = feature_extractor
        else:
            cache_path = data_path.replace('.csv', '_daily_features_v3.pkl')
            self.fe = DailyFeatureExtractorV3(data_path, cache_path=cache_path)

        # V3: 计算动态分位数权重所需的经验CDF
        if self.normalize and not self.fe._dynamic_weight_available:
            self.fe.compute_position_quantiles(self.data_normalized)

        # Full window set
        self.num_samples = len(self.data_normalized) - self.seq_len + 1

        # Apply split filter if requested
        self.split = split
        if split is not None:
            train_days, val_days, test_days = get_monthly_stratified_split(data_path)
            if split == 'train':
                allowed_days = set(train_days.tolist())
            elif split == 'val':
                allowed_days = set(val_days.tolist())
            elif split == 'test':
                allowed_days = set(test_days.tolist())
            else:
                raise ValueError(f"Unknown split '{split}', expected 'train'/'val'/'test'")
            self._allowed_indices = [i for i in range(self.num_samples)
                                      if (i // 24) in allowed_days]
        else:
            self._allowed_indices = list(range(self.num_samples))

    def __len__(self):
        return len(self._allowed_indices)

    def __getitem__(self, idx):
        real_idx = self._allowed_indices[idx]
        sequence = self.data_normalized[real_idx: real_idx + self.seq_len]
        sequence = torch.FloatTensor(sequence).permute(1, 0)
        day_idx = real_idx // 24
        condition = torch.FloatTensor(self.fe.get_condition(day_idx))
        node_feat = torch.FloatTensor(self.fe.get_node_features(day_idx))
        return sequence, condition, node_feat

    def denormalize(self, normalized_tensor):
        if self.scaler is None:
            return normalized_tensor
        np_arr = normalized_tensor.detach().permute(1, 0).cpu().numpy()
        denormed = self.scaler.inverse_transform(np_arr)
        return torch.FloatTensor(denormed).permute(1, 0)

    def get_dynamic_tail_weight(self, x_normalized):
        """获取动态分位数尾部权重, 传给扩散模型loss"""
        return self.fe.get_dynamic_tail_weight(x_normalized)


if __name__ == '__main__':
    import os as _os
    DATA_PATH = "./源荷数据集.csv"

    # 删除旧V3缓存测试
    cache = DATA_PATH.replace('.csv', '_daily_features_v3.pkl')
    if _os.path.exists(cache):
        _os.remove(cache)

    # Test split
    train_days, val_days, test_days = get_monthly_stratified_split(DATA_PATH)
    print(f"Train days: {len(train_days)}, Val days: {len(val_days)}, Test days: {len(test_days)}")
    print(f"Test days: {test_days.tolist()}")

    fe = DailyFeatureExtractorV3(DATA_PATH)
    fe.print_summary()

    ds = EnergyDataset1DV3(DATA_PATH, seq_len=24, feature_extractor=fe)
    print(f"Full dataset size: {len(ds)}")

    ds_train = EnergyDataset1DV3(DATA_PATH, seq_len=24, feature_extractor=fe, split='train')
    ds_val = EnergyDataset1DV3(DATA_PATH, seq_len=24, feature_extractor=fe, split='val')
    ds_test = EnergyDataset1DV3(DATA_PATH, seq_len=24, feature_extractor=fe, split='test')
    print(f"Train windows: {len(ds_train)}, Val windows: {len(ds_val)}, Test windows: {len(ds_test)}")

    seq, cond, node = ds[0]
    print(f"序列形状: {seq.shape}, 条件向量形状: {cond.shape}, 节点特征: {node.shape}")

    # Test day lookup
    jan1_idx = get_test_day_index(DATA_PATH, 1, 1)
    jul1_idx = get_test_day_index(DATA_PATH, 7, 1)
    print(f"Jan 1 day index: {jan1_idx}, Jul 1 day index: {jul1_idx}")

    # 测试EVT引导条件构造
    print("\n=== EVT引导条件构造测试 ===")
    for level in [0.0, 0.5, 0.9, 0.95, 0.99]:
        cond = fe.build_extreme_condition(180, extreme_level=level)
        print(f"Level {level:.2f}: EVT超越={cond[26:31]}, 尾部概率={cond[31:36]}, "
              f"极端标识={cond[36]:.0f}")

    # 测试per-variable控制
    print("\n逐变量控制 (wind P99):")
    cond = fe.build_extreme_condition(180, extreme_level=0.0,
                                       per_variable={'wind': 0.99})
    print(f"  EVT超越={cond[26:31]}, 尾部概率={cond[31:36]}")

    # 测试动态权重
    x_test = torch.rand(2, 5, 24)
    w = ds.get_dynamic_tail_weight(x_test)
    print(f"\n动态尾部权重: shape={w.shape}, 范围=[{w.min():.3f}, {w.max():.3f}]")
