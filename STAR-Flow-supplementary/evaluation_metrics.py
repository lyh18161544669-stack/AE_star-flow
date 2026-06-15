"""
Evaluation metrics for energy scenario generation quality.

Metrics:
  1. CRPS (Continuous Ranked Probability Score) — per-variable, per-time-step
  2. Energy Distance — overall distribution distance
  3. Sliced Wasserstein Distance — efficient distribution distance

All metrics operate on normalized data in [0, 1].

References:
  - CRPS: Gneiting & Raftery (2007), JASA
  - Energy Distance: Székely & Rizzo (2013), J. Stat. Planning & Inference
  - SWD: Rabin et al. (2011), Wasserstein Barycenter and Its Application
  - CDSG: Zhao et al. (2025), Applied Energy 377, 124555
"""

import numpy as np
from scipy.spatial.distance import cdist
from scipy.stats import wasserstein_distance

# ---- CRPS (Continuous Ranked Probability Score) ----

def crps_sample(y_pred, y_true):
    """CRPS for a single observation against an ensemble of predictions.

    Uses the sample-based energy form (Gneiting & Raftery, 2007):
      CRPS(F, x) = E[|X - x|] - 0.5 * E[|X - X'|]

    Args:
        y_pred: (N,) array of N ensemble predictions (generated scenarios)
        y_true: scalar, the true observation

    Returns:
        scalar CRPS value
    """
    N = len(y_pred)
    if N == 0:
        return 0.0
    # E[|X - x|]
    term1 = np.mean(np.abs(y_pred - y_true))
    if N == 1:
        return term1
    # E[|X - X'|] — all pairwise absolute differences
    # Vectorized: use pdist or compute via broadcasting the upper triangle
    y_sorted = np.sort(y_pred)
    # Pairwise mean absolute diff = 2/(N(N-1)) * Σ_{i<j} |y_i - y_j|
    # For sorted array: Σ_{i<j} (y_j - y_i) = Σ_k (2k-N-1) * y_k
    weights = 2 * np.arange(N) - N + 1  # (2k - N - 1)
    pairwise_sum = np.dot(weights, y_sorted)
    # Mean over N*(N-1)/2 pairs, then divide by N^2 for the unbiased estimator
    pairwise_mean = pairwise_sum / (N * (N - 1) / 2) if N > 1 else 0.0
    return term1 - 0.5 * pairwise_mean


def compute_crps_all(generated, real):
    """Compute CRPS per variable, per time step, and aggregated.

    Args:
        generated: (N, C, T) array — N generated scenarios, C channels, T time steps
        real: (C, T) array — one real day

    Returns:
        dict with keys:
          'per_variable': (C,) array of per-variable mean CRPS
          'per_time': (T,) array of per-time-step mean CRPS (across variables)
          'crps_sum': scalar CRPS for the sum of all variables (CRPS-sum)
          'mean': scalar overall mean CRPS
    """
    N, C, T = generated.shape
    crps_vt = np.zeros((C, T))
    for c in range(C):
        for t in range(T):
            crps_vt[c, t] = crps_sample(generated[:, c, t], real[c, t])

    # CRPS-sum: CRPS of the distribution of sum of all variables at each time step
    gen_sum = generated.sum(axis=1)  # (N, T)
    real_sum = real.sum(axis=0)      # (T,)
    crps_sum_t = np.zeros(T)
    for t in range(T):
        crps_sum_t[t] = crps_sample(gen_sum[:, t], real_sum[t])

    return {
        'per_variable': crps_vt.mean(axis=1),       # (C,)
        'per_time': crps_vt.mean(axis=0),            # (T,)
        'crps_sum': crps_sum_t.mean(),               # scalar
        'mean': crps_vt.mean(),                      # scalar
        'matrix': crps_vt,                           # (C, T) full matrix
    }


def compute_crps_normalized(generated, real):
    """CRPS normalized by mean absolute real value (as in CDSG paper Eq. 39).

    normalized_CRPS = Σ_{c,t} CRPS(c,t) / Σ_{c,t} |real(c,t)|
    """
    crps = compute_crps_all(generated, real)
    total_real = np.abs(real).sum()
    if total_real > 0:
        crps['normalized'] = crps['matrix'].sum() / total_real
    else:
        crps['normalized'] = 0.0
    return crps


# ---- Energy Distance ----

def energy_distance(X, Y):
    """Energy distance between two sample sets (Székely & Rizzo, 2013).

    D(X, Y) = 2 * E||X-Y|| - E||X-X'|| - E||Y-Y'||

    D(X, Y) >= 0, and D(X, Y) = 0 iff X and Y are identically distributed.
    This is a proper metric (satisfies triangle inequality).

    Args:
        X: (n, d) array — real samples (n samples, d dimensions)
        Y: (m, d) array — generated samples (m samples, d dimensions)

    Returns:
        scalar energy distance (>= 0)
    """
    n = X.shape[0]
    m = Y.shape[0]

    # Cross-term: E||X-Y|| — mean over all n*m pairs
    cross_dists = cdist(X, Y, metric='euclidean')  # (n, m)
    cross = cross_dists.mean()

    # X self-term: E||X-X'|| — mean over n*(n-1) distinct pairs
    if n > 1:
        xx_dists = cdist(X, X, metric='euclidean')
        xx = xx_dists.sum() / (n * (n - 1))  # exclude diagonal (zeros)
    else:
        xx = 0.0

    # Y self-term: E||Y-Y'|| — mean over m*(m-1) distinct pairs
    if m > 1:
        yy_dists = cdist(Y, Y, metric='euclidean')
        yy = yy_dists.sum() / (m * (m - 1))
    else:
        yy = 0.0

    return 2.0 * cross - xx - yy


def compute_energy_distance_per_time(generated, real_days):
    """Energy distance computed per time step (5-dim vector per hour).

    Args:
        generated: (N, C, T) — N generated scenarios
        real_days: (D, C, T) — D real days

    Returns:
        dict with per_time and mean energy distance
    """
    N, C, T = generated.shape
    D = real_days.shape[0]

    ed_per_t = np.zeros(T)
    for t in range(T):
        # At time t: each sample is a C-dim vector
        X_t = real_days[:, :, t]     # (D, C)
        Y_t = generated[:, :, t]     # (N, C)
        ed_per_t[t] = energy_distance(X_t, Y_t)

    return {
        'per_time': ed_per_t,
        'mean': ed_per_t.mean(),
    }


def compute_energy_distance_full(generated, real_days):
    """Energy distance on the full day vector (C*T dim, flattened).

    Args:
        generated: (N, C, T) — N generated scenarios
        real_days: (D, C, T) — D real days

    Returns:
        scalar energy distance on full-day distribution
    """
    N, C, T = generated.shape
    X = real_days.reshape(real_days.shape[0], -1)   # (D, C*T)
    Y = generated.reshape(N, -1)                     # (N, C*T)
    return energy_distance(X, Y)


# ---- Sliced Wasserstein Distance ----

def sliced_wasserstein_distance(X, Y, n_projections=200, seed=42):
    """Sliced Wasserstein Distance via random 1D projections.

    SWD(X, Y) = E_{θ~U(S^{d-1})}[W_1(P_θ(X), P_θ(Y))]

    where W_1 is the 1-D Wasserstein distance (Earth Mover's Distance)
    computed via scipy.stats.wasserstein_distance.

    Args:
        X: (n, d) array — real samples
        Y: (m, d) array — generated samples
        n_projections: number of random projection directions
        seed: random seed for reproducibility

    Returns:
        scalar SWD value
    """
    rng = np.random.RandomState(seed)
    n, d = X.shape
    swd_vals = []

    for _ in range(n_projections):
        # Random direction on the unit sphere
        theta = rng.randn(d)
        theta = theta / np.linalg.norm(theta)

        # Project both sets onto theta
        x_proj = X.dot(theta)
        y_proj = Y.dot(theta)

        # 1D Wasserstein distance
        w1 = wasserstein_distance(x_proj, y_proj)
        swd_vals.append(w1)

    return np.mean(swd_vals)


def compute_swd_per_time(generated, real_days, n_projections=200):
    """Sliced Wasserstein Distance per time step (5-dim vectors)."""
    N, C, T = generated.shape
    D = real_days.shape[0]
    swd_per_t = np.zeros(T)
    for t in range(T):
        X_t = real_days[:, :, t]
        Y_t = generated[:, :, t]
        swd_per_t[t] = sliced_wasserstein_distance(X_t, Y_t, n_projections=n_projections)
    return {'per_time': swd_per_t, 'mean': swd_per_t.mean()}


def compute_swd_full(generated, real_days, n_projections=200):
    """Sliced Wasserstein Distance on full-day vectors (C*T dim, flattened)."""
    N, C, T = generated.shape
    X = real_days.reshape(real_days.shape[0], -1)
    Y = generated.reshape(N, -1)
    return sliced_wasserstein_distance(X, Y, n_projections=n_projections)


# ---- Convenience: all distribution metrics in one call ----

def evaluate_distribution_quality(generated, real_day, real_dataset,
                                   real_subset=None, n_projections=200):
    """Compute all distribution quality metrics for one extreme level.

    Args:
        generated: (N, C, T) — normalized, N generated scenarios
        real_day: (C, T) — normalized, the template real day (e.g., day 180)
        real_dataset: (D, C, T) — normalized, set of real days (e.g., 365 days)
                      used as reference distribution for energy distance / SWD
        real_subset: (S, C, T) or None — subset of real days for per-time metrics
                     (e.g., same-season days). If None, uses all real_dataset.
        n_projections: SWD projections

    Returns:
        dict of all metrics
    """
    if real_subset is None:
        real_subset = real_dataset

    results = {}

    # CRPS — against the template day
    crps = compute_crps_all(generated, real_day)
    results['crps'] = crps

    # Energy Distance — per time and full day
    ed_time = compute_energy_distance_per_time(generated, real_subset)
    ed_full = compute_energy_distance_full(generated, real_subset)
    results['energy_distance_per_time'] = ed_time
    results['energy_distance_full'] = ed_full

    # Sliced Wasserstein Distance
    swd_time = compute_swd_per_time(generated, real_subset, n_projections)
    swd_full = compute_swd_full(generated, real_subset, n_projections)
    results['swd_per_time'] = swd_time
    results['swd_full'] = swd_full

    return results


# ---- Statistical moment metrics ----

def compute_moment_errors(generated, real_dataset):
    """Compute errors in first 4 moments (expectation, variance, skewness, kurtosis).

    Args:
        generated: (N, C, T) — N generated scenarios
        real_dataset: (D, C, T) — D real days

    Returns:
        dict with per-variable absolute errors for each moment
    """
    from scipy.stats import skew, kurtosis
    N, C, T = generated.shape
    D = real_dataset.shape[0]

    # Flatten real data (across days and time) per variable
    real_flat = real_dataset.transpose(1, 0, 2).reshape(C, -1)  # (C, D*T)
    gen_flat = generated.transpose(1, 0, 2).reshape(C, -1)      # (C, N*T)

    moments = {}
    for c in range(C):
        r = real_flat[c]
        g = gen_flat[c]

        moments[f'var_{c}'] = {
            'mean_err': abs(g.mean() - r.mean()),
            'std_err': abs(g.std() - r.std()),
            'skew_err': abs(skew(g) - skew(r)),
            'kurt_err': abs(kurtosis(g) - kurtosis(r)),
        }

    # Aggregate across variables
    moments['avg'] = {
        'mean_err': np.mean([moments[f'var_{c}']['mean_err'] for c in range(C)]),
        'std_err': np.mean([moments[f'var_{c}']['std_err'] for c in range(C)]),
        'skew_err': np.mean([moments[f'var_{c}']['skew_err'] for c in range(C)]),
        'kurt_err': np.mean([moments[f'var_{c}']['kurt_err'] for c in range(C)]),
    }

    return moments


# ---- Correlation matrix error ----

def compute_correlation_error(generated, real_day):
    """Compute Pearson correlation matrix error (Frobenius norm).

    Args:
        generated: (N, C, T)
        real_day: (C, T)

    Returns:
        dict with per-sample corr error and mean
    """
    N, C, T = generated.shape
    real_corr = np.corrcoef(real_day)  # (C, C)

    errors = []
    for i in range(N):
        gen_corr = np.corrcoef(generated[i])  # (C, C)
        err = np.linalg.norm(gen_corr - real_corr, ord='fro')
        errors.append(err)

    return {
        'per_sample': np.array(errors),
        'mean': np.mean(errors),
        'std': np.std(errors),
        'real_corr': real_corr,
    }


# ---- Autocorrelation error ----

def compute_autocorr_error(generated, real_day, max_lag=6):
    """Compute autocorrelation error per variable.

    Args:
        generated: (N, C, T)
        real_day: (C, T)
        max_lag: maximum lag for autocorrelation

    Returns:
        dict with per-variable per-lag errors
    """
    N, C, T = generated.shape
    errors = np.zeros((C, max_lag))

    for c in range(C):
        real_acf = _autocorr(real_day[c], max_lag)
        gen_acfs = np.zeros((N, max_lag))
        for i in range(N):
            gen_acfs[i] = _autocorr(generated[i, c], max_lag)
        gen_acf_mean = gen_acfs.mean(axis=0)
        errors[c] = np.abs(gen_acf_mean - real_acf)

    return {
        'per_variable_per_lag': errors,
        'mean': errors.mean(),
        'per_variable_mean': errors.mean(axis=1),
    }


def _autocorr(x, max_lag):
    """Compute autocorrelation for lags 1..max_lag."""
    n = len(x)
    x_centered = x - x.mean()
    denom = np.sum(x_centered ** 2)
    if denom == 0:
        return np.zeros(max_lag)
    acf = np.zeros(max_lag)
    for lag in range(1, max_lag + 1):
        acf[lag - 1] = np.sum(x_centered[lag:] * x_centered[:-lag]) / denom
    return acf


# ---- Tail Calibration Metrics ----
# These are essential for evaluating extreme-level scenario generation.
# Normal CRPS is NOT meaningful for extreme scenarios (they deliberately
# deviate from any single reference day). Instead, we verify that:
#   1. The generated P95/P99 scenarios match the claimed percentiles of
#      the real data distribution (tail calibration).
#   2. The generated extreme scenarios cover real extreme events
#      (extreme coverage rate).
#   3. The distribution quality specifically in the tail region is good
#      (CRPS-tail / tail Wasserstein distance).
#
# References:
#   - TailDiff: Naumov et al. (ICML 2024) — tail calibration for diffusion
#   - Ren et al. (NeurIPS 2024) — importance weighting for tail recovery
#   - CDSG: Zhao et al. (2025), Applied Energy 377, 124555


def compute_tail_calibration(generated, real_data, target_percentile=95):
    """Tail calibration error: does the median of generated extreme scenarios
    match the claimed percentile of the real data distribution?

    For extreme_level=0.95 scenarios, the median daily max of generated
    scenarios should be close to the P95 daily max of real data.

    Args:
        generated: (N, C, T) — extreme-level generated scenarios
        real_data: (D, C, T) — all real days (365)
        target_percentile: int — 95 for 'extreme', 99 for 'severe'

    Returns:
        dict with:
          'per_variable': (C,) normalized absolute error per variable
          'mean': scalar mean calibration error across variables
          'gen_median': (C,) median daily max of generated
          'real_target': (C,) target percentile of real daily max
    """
    C = generated.shape[1]
    real_daily_max = real_data.max(axis=2)   # (D, C)
    gen_daily_max = generated.max(axis=2)     # (N, C)

    errors = np.zeros(C)
    gen_medians = np.zeros(C)
    real_targets = np.zeros(C)

    for c in range(C):
        gm = np.median(gen_daily_max[:, c])
        rt = np.percentile(real_daily_max[:, c], target_percentile)
        gen_medians[c] = gm
        real_targets[c] = rt
        errors[c] = abs(gm - rt) / max(rt, 1e-8)

    return {
        'per_variable': errors,
        'mean': errors.mean(),
        'gen_median': gen_medians,
        'real_target': real_targets,
    }


def compute_extreme_coverage_rate(generated, real_data, percentile=95,
                                    level_name=None):
    """Extreme coverage rate: do generated scenarios cover the right
    proportion of real extreme events?

    For normal scenarios (level_name='normal'):
      ~5% of generated daily max values should exceed real P95.
      Coverage ratio ≈ 1.0 is ideal.

    For extreme scenarios (level_name='high_risk'/'extreme'):
      Generated scenarios should have a MUCH higher extreme rate.
      Coverage ratio should be >> 1.0 (no single ideal value).
      This measures how aggressively the model pushes into the tail.

    Args:
        generated: (N, C, T) — scenarios at this level
        real_data: (D, C, T) — all real days
        percentile: int — the extreme threshold in real data
        level_name: str or None — 'normal', 'high_risk', or 'extreme'

    Returns:
        dict with:
          'per_variable': (C,) coverage ratio per variable
          'mean': mean coverage ratio
          'ideal': 1.0 for normal, None for extreme (no single ideal)
    """
    C = generated.shape[1]
    real_daily_max = real_data.max(axis=2)   # (D, C)
    gen_daily_max = generated.max(axis=2)     # (N, C)

    ideal_rate = 1.0 - percentile / 100.0

    coverage = np.zeros(C)
    for c in range(C):
        threshold = np.percentile(real_daily_max[:, c], percentile)
        gen_extreme_rate = (gen_daily_max[:, c] >= threshold).mean()
        coverage[c] = gen_extreme_rate / max(ideal_rate, 1e-8)

    ideal_value = 1.0 if (level_name == 'normal') else None

    return {
        'per_variable': coverage,
        'mean': coverage.mean(),
        'ideal': ideal_value,
    }


def compute_crps_tail(generated, real_data, tail_pct=90):
    """Tail CRPS: distribution distance evaluated only on tail time steps.

    Uses 1D Wasserstein distance between generated and real values
    at time steps where real data exceeds the tail_pct threshold.

    Args:
        generated: (N, C, T)
        real_data: (D, C, T)
        tail_pct: percentile threshold for "tail" region

    Returns:
        dict with:
          'per_variable': (C,) tail Wasserstein distance per variable
          'mean': scalar mean across variables
    """
    N, C, T = generated.shape
    D = real_data.shape[0]

    tail_wd = np.zeros(C)
    for c in range(C):
        thresholds = np.percentile(real_data[:, c, :], tail_pct, axis=0)  # (T,)

        # Real data: only tail values (above threshold at each time step)
        real_tail_vals = []
        for d in range(D):
            mask = real_data[d, c, :] >= thresholds
            if mask.any():
                real_tail_vals.extend(real_data[d, c, mask].tolist())

        # Generated data: also only tail values (same thresholds)
        gen_tail_vals = []
        for n in range(N):
            mask = generated[n, c, :] >= thresholds
            if mask.any():
                gen_tail_vals.extend(generated[n, c, mask].tolist())

        if len(real_tail_vals) > 10 and len(gen_tail_vals) > 10:
            from scipy.stats import wasserstein_distance
            tail_wd[c] = wasserstein_distance(gen_tail_vals, real_tail_vals)
        else:
            tail_wd[c] = 0.0

    return {
        'per_variable': tail_wd,
        'mean': tail_wd.mean(),
    }


# ---- PIT (Probability Integral Transform) ----

def compute_pit_histogram(generated, real, n_bins=10):
    """Marginal PIT histogram for calibration diagnostics.

    For each (c, t) cell, compute empirical CDF from N generated scenarios,
    evaluate at the real observation. Pool all cells to form PIT histogram.

    Interpretation:
      Uniform(0,1) → well calibrated
      U-shaped      → underdispersed (predictive intervals too narrow)
      Hump-shaped   → overdispersed (predictive intervals too wide)
      Skewed        → systematic bias

    Args:
        generated: (N, C, T) — N generated scenarios
        real: (C, T) — one real day
        n_bins: number of histogram bins (default 10)

    Returns:
        dict with: 'pit_values' (C*T,), 'hist_counts' (n_bins,),
                   'bin_edges' (n_bins+1,), 'ks_statistic', 'chi2_statistic'
    """
    N, C, T = generated.shape
    pit_values = np.zeros(C * T)
    idx = 0

    for c in range(C):
        for t in range(T):
            gen_vals = generated[:, c, t]
            real_val = real[c, t]
            # Empirical CDF: fraction of generated <= real
            pit_values[idx] = (gen_vals <= real_val).mean()
            idx += 1

    # Histogram
    hist_counts, bin_edges = np.histogram(pit_values, bins=n_bins, range=(0, 1))

    # Kolmogorov-Smirnov test against Uniform(0,1)
    from scipy.stats import kstest
    ks_stat, _ = kstest(pit_values, 'uniform')

    # Chi-squared uniformity test
    expected = len(pit_values) / n_bins
    chi2 = np.sum((hist_counts - expected) ** 2 / max(expected, 1e-8))

    return {
        'pit_values': pit_values,
        'hist_counts': hist_counts,
        'bin_edges': bin_edges,
        'ks_statistic': float(ks_stat),
        'chi2_statistic': float(chi2),
        'uniform_expected': float(expected),
    }


# ---- Precision / Recall (Sajjadi et al. 2018) ----

def _knn_manifold_radius(X, k=5):
    """Compute k-th nearest neighbor distance for each sample in X.

    Args:
        X: (N, D) — N samples, D dimensions
        k: neighbor index (default 5)

    Returns:
        (N,) array of distances to k-th nearest neighbor
    """
    from scipy.spatial.distance import cdist
    dists = cdist(X, X, metric='euclidean')  # (N, N)
    # Sort each row, take k-th (skip self at index 0)
    sorted_dists = np.sort(dists, axis=1)
    return sorted_dists[:, min(k, dists.shape[1] - 1)]


def compute_precision_recall_g(real_data, generated, k=5,
                                 n_components=20, seed=42):
    """precision_g and recall_g via manifold overlap (Sajjadi et al. 2018).

    Uses PCA dimensionality reduction to avoid the curse of dimensionality
    in high-dimensional spaces (e.g., 5x24=120 dims with only 365 samples).

    precision_g = fraction of generated samples inside real data manifold
    recall_g    = fraction of real samples inside generated manifold

    manifold(X) = ∪_i B(X_i, NND_k(X_i))
    where NND_k(X_i) = distance to k-th nearest neighbor of X_i in X.

    Args:
        real_data: (D, F) — D real samples flattened to F dimensions
        generated: (N, F) — N generated samples flattened to F dimensions
        k: k-th nearest neighbor for manifold radius (default 5)
        n_components: PCA target dimensions (default 20, 0 = no PCA)
        seed: random seed for PCA

    Returns:
        dict with: 'precision', 'recall', 'n_components',
                   'explained_variance', 'precision_per_sample', 'recall_per_sample'
    """
    from sklearn.decomposition import PCA

    D = len(real_data)
    N = len(generated)

    # PCA dimensionality reduction
    if n_components > 0 and n_components < real_data.shape[1]:
        all_data = np.vstack([real_data, generated])
        n_comp = min(n_components, all_data.shape[1], all_data.shape[0] // 5)
        pca = PCA(n_components=n_comp, random_state=seed)
        pca.fit(all_data)
        real_low = pca.transform(real_data)
        gen_low = pca.transform(generated)
        explained_var = float(pca.explained_variance_ratio_.sum())
    else:
        real_low = real_data
        gen_low = generated
        n_comp = real_data.shape[1]
        explained_var = 1.0

    # Compute manifold radii in low-dimensional space
    from scipy.spatial.distance import cdist
    real_radii = _knn_manifold_radius(real_low, k=k)   # (D,)
    gen_radii = _knn_manifold_radius(gen_low, k=k)     # (N,)

    # Cross-distance matrices
    gen_to_real = cdist(gen_low, real_low, metric='euclidean')  # (N, D)
    real_to_gen = cdist(real_low, gen_low, metric='euclidean')  # (D, N)

    # precision_g: for each generated sample, is it within some real sphere?
    gen_in_real = np.zeros(N, dtype=bool)
    for i in range(N):
        gen_in_real[i] = np.any(gen_to_real[i] <= real_radii)

    # recall_g: for each real sample, is it within some generated sphere?
    real_in_gen = np.zeros(D, dtype=bool)
    for j in range(D):
        real_in_gen[j] = np.any(real_to_gen[j] <= gen_radii)

    return {
        'precision': float(gen_in_real.mean()),
        'recall': float(real_in_gen.mean()),
        'n_components': n_comp,
        'explained_variance': explained_var,
        'precision_per_sample': gen_in_real,
        'recall_per_sample': real_in_gen,
    }


# ---- Coverage Rate (CR) / Interval Width (IW) ----

def compute_cr_iw(generated, real_samples, alpha=0.05):
    """Coverage Rate and Interval Width at (1-alpha) confidence level.

    For each (c, t) cell, compute [L, U] from N generated scenarios.
    CR = fraction of real observations falling within [L, U].
    IW = mean interval width U - L.

    Ideal: CR ≈ (1-alpha) with narrow IW.

    Args:
        generated: (N, C, T) — N generated scenarios
        real_samples: (D, C, T) or (C, T) — real data
                      If (D, C, T): compute CR/IW per real day, then average
        alpha: significance level (default 0.05 → 95% confidence)

    Returns:
        dict with: 'cr', 'iw', 'cr_per_variable', 'iw_per_variable'
    """
    N, C, T = generated.shape

    # Predictive intervals from generated scenarios
    lower = np.percentile(generated, 100 * alpha / 2, axis=0)   # (C, T)
    upper = np.percentile(generated, 100 * (1 - alpha / 2), axis=0)  # (C, T)

    # Normalize real_samples to (D, C, T) layout
    if real_samples.ndim == 3:
        if real_samples.shape[2] == C and real_samples.shape[1] != C:
            # (D, T, C) → (D, C, T)
            real_samples = real_samples.transpose(0, 2, 1)
        D = real_samples.shape[0]

        cr_per_var = np.zeros(C)
        iw_per_var = np.zeros(C)

        for c in range(C):
            # Vectorized: for all D days and T time steps at once
            real_c = real_samples[:, c, :]  # (D, T)
            covered = (real_c >= lower[c]) & (real_c <= upper[c])
            cr_per_var[c] = covered.mean()

            iw_per_var[c] = (upper[c] - lower[c]).mean()

        return {
            'cr': cr_per_var.mean(),
            'iw': iw_per_var.mean(),
            'cr_per_variable': cr_per_var,
            'iw_per_variable': iw_per_var,
        }
    else:
        # Single real day: check layout
        # (C, T) → shape[0]==C, shape[1]==T
        # (T, C) → shape[0]==T, shape[1]==C
        if real_samples.shape[0] == C:
            real = real_samples  # already (C, T)
        else:
            real = real_samples.T  # (T, C) → (C, T)

        cr_per_var = np.zeros(C)
        iw_per_var = np.zeros(C)
        for c in range(C):
            covered = (real[c] >= lower[c]) & (real[c] <= upper[c])
            cr_per_var[c] = covered.mean()
            iw_per_var[c] = (upper[c] - lower[c]).mean()

        return {
            'cr': cr_per_var.mean(),
            'iw': iw_per_var.mean(),
            'cr_per_variable': cr_per_var,
            'iw_per_variable': iw_per_var,
        }


# ==============================================================================
# Self-test
# ==============================================================================
if __name__ == '__main__':
    print("=" * 60)
    print("Evaluation Metrics Self-Test")
    print("=" * 60)

    rng = np.random.RandomState(42)

    # Generate synthetic data
    N_gen = 100
    N_real = 365
    C = 5
    T = 24

    # Real data: sinusoidal + noise
    t = np.linspace(0, 2 * np.pi, T)
    real_days = np.zeros((N_real, C, T))
    for d in range(N_real):
        for c in range(C):
            phase = rng.randn() * 0.2
            amp = 0.3 + 0.1 * c + rng.randn() * 0.05
            offset = 0.3 + 0.05 * c + rng.randn() * 0.02
            real_days[d, c] = np.clip(
                offset + amp * np.sin(t + phase) + rng.randn(T) * 0.05,
                0.0, 1.0
            )

    real_day = real_days[180]

    # Generated data: same distribution + small shift
    shift = 0.02
    generated_good = np.clip(real_days[:N_gen] + shift + rng.randn(N_gen, C, T) * 0.03, 0, 1)

    # Generated bad: random noise
    generated_bad = rng.rand(N_gen, C, T)

    print("\n[1] CRPS")
    crps_good = compute_crps_all(generated_good, real_day)
    crps_bad = compute_crps_all(generated_bad, real_day)
    print(f"  Good model CRPS mean: {crps_good['mean']:.6f}")
    print(f"  Bad model CRPS mean:  {crps_bad['mean']:.6f}")
    print(f"  Ratio bad/good: {crps_bad['mean'] / crps_good['mean']:.1f}x "
          f"{'PASS' if crps_bad['mean'] > crps_good['mean'] else 'FAIL'}")
    print(f"  CRPS-sum good: {crps_good['crps_sum']:.6f}")
    print(f"  CRPS-sum bad:  {crps_bad['crps_sum']:.6f}")

    print("\n[2] Energy Distance")
    ed_good = energy_distance(
        real_days.reshape(N_real, -1),
        generated_good.reshape(N_gen, -1)
    )
    ed_bad = energy_distance(
        real_days.reshape(N_real, -1),
        generated_bad.reshape(N_gen, -1)
    )
    print(f"  Good model ED: {ed_good:.6f}")
    print(f"  Bad model ED:  {ed_bad:.6f}")
    print(f"  Ratio bad/good: {ed_bad / ed_good:.1f}x "
          f"{'PASS' if ed_bad > ed_good else 'FAIL'}")

    print("\n[3] Sliced Wasserstein Distance")
    swd_good = sliced_wasserstein_distance(
        real_days.reshape(N_real, -1),
        generated_good.reshape(N_gen, -1),
        n_projections=100
    )
    swd_bad = sliced_wasserstein_distance(
        real_days.reshape(N_real, -1),
        generated_bad.reshape(N_gen, -1),
        n_projections=100
    )
    print(f"  Good model SWD: {swd_good:.6f}")
    print(f"  Bad model SWD:  {swd_bad:.6f}")
    print(f"  Ratio bad/good: {swd_bad / swd_good:.1f}x "
          f"{'PASS' if swd_bad > swd_good else 'FAIL'}")

    print("\n[4] Moment Errors")
    moments_good = compute_moment_errors(generated_good, real_days)
    moments_bad = compute_moment_errors(generated_bad, real_days)
    print(f"  Good avg mean_err={moments_good['avg']['mean_err']:.4f} "
          f"std_err={moments_good['avg']['std_err']:.4f}")
    print(f"  Bad  avg mean_err={moments_bad['avg']['mean_err']:.4f} "
          f"std_err={moments_bad['avg']['std_err']:.4f}")

    print("\n[5] Correlation Error")
    corr_good = compute_correlation_error(generated_good, real_day)
    corr_bad = compute_correlation_error(generated_bad, real_day)
    print(f"  Good corr err: {corr_good['mean']:.4f}")
    print(f"  Bad  corr err: {corr_bad['mean']:.4f}")

    print("\n[6] Energy Distance — proper metric property")
    # Energy distance of a distribution to itself should be ~0
    ed_self = energy_distance(
        real_days[:50].reshape(50, -1),
        real_days[50:100].reshape(50, -1)
    )
    print(f"  ED(real, real): {ed_self:.6f} (should be ~0)")
    print(f"  {'PASS' if ed_self < 0.1 else 'FAIL (unexpectedly large)'}")

    print("\n[7] PIT Histogram")
    pit = compute_pit_histogram(generated_good, real_day)
    print(f"  Good model PIT: KS={pit['ks_statistic']:.4f}, chi2={pit['chi2_statistic']:.1f}")
    print(f"  Hist counts: {pit['hist_counts']}")
    print(f"  {'PASS' if pit['ks_statistic'] < 0.3 else 'MARGINAL'}")

    pit_bad = compute_pit_histogram(generated_bad, real_day)
    print(f"  Bad model PIT:  KS={pit_bad['ks_statistic']:.4f}, chi2={pit_bad['chi2_statistic']:.1f}")
    print(f"  {'PASS' if pit_bad['ks_statistic'] > pit['ks_statistic'] else 'MARGINAL'}")

    print("\n[8] Precision/Recall (Sajjadi et al.)")
    X_real = real_days.reshape(N_real, -1)
    Y_good = generated_good.reshape(N_gen, -1)
    Y_bad = generated_bad.reshape(N_gen, -1)
    pr_good = compute_precision_recall_g(X_real, Y_good)
    pr_bad = compute_precision_recall_g(X_real, Y_bad)
    print(f"  Good: precision={pr_good['precision']:.4f}, recall={pr_good['recall']:.4f}")
    print(f"  Bad:  precision={pr_bad['precision']:.4f}, recall={pr_bad['recall']:.4f}")
    print(f"  {'PASS' if pr_good['recall'] > pr_bad['recall'] else 'OK (good model may have lower rec if conservative)'}")

    print("\n[9] Coverage Rate / Interval Width")
    cr_good = compute_cr_iw(generated_good, real_day)
    cr_bad = compute_cr_iw(generated_bad, real_day)
    print(f"  Good: CR_95={cr_good['cr']:.4f} ({cr_good['cr']*100:.1f}%), IW={cr_good['iw']:.4f}")
    print(f"  Bad:  CR_95={cr_bad['cr']:.4f} ({cr_bad['cr']*100:.1f}%), IW={cr_bad['iw']:.4f}")
    print(f"  {'PASS' if abs(cr_good['cr'] - 0.95) < abs(cr_bad['cr'] - 0.95) else 'MARGINAL'}")

    print("\n[10] Tail Calibration & Extreme Coverage")
    tc = compute_tail_calibration(generated_good, real_days, target_percentile=95)
    ec = compute_extreme_coverage_rate(generated_good, real_days, percentile=95)
    wt = compute_crps_tail(generated_good, real_days, tail_pct=90)
    print(f"  Tail calib error: {tc['mean']:.4f}")
    print(f"  Ext coverage ratio: {ec['mean']:.4f} (ideal=1.0)")
    print(f"  CRPS-tail (WD): {wt['mean']:.4f}")
    print(f"  {'PASS' if tc['mean'] < 0.5 else 'MARGINAL'}")

    print(f"\n{'='*60}")
    print("All tests completed!")
