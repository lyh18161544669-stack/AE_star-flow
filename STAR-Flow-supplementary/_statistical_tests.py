"""Statistical significance testing for model comparison.

Uses paired Wilcoxon signed-rank test on 12 per-day metric values,
with Benjamini-Hochberg FDR correction for multiple comparisons.

Design rationale:
  - Unit of analysis: 12 test days (paired), NOT individual scenarios
  - Wilcoxon: non-parametric, robust to non-normal metric distributions
  - FDR correction: controls false discovery rate across all pairwise tests

Reference: CDSG (Zhao et al. 2025, Applied Energy) uses paired t-test;
  we prefer Wilcoxon for its weaker distributional assumptions.
"""
import numpy as np
from scipy.stats import wilcoxon
from scipy.stats import mannwhitneyu as _mannwhitneyu  # only for independent-sample ref


def pairwise_wilcoxon(per_day_values, model_names, alpha=0.05):
    """Run all pairwise Wilcoxon signed-rank tests with FDR correction.

    Args:
        per_day_values: dict {model_name: (D,) array} of per-day metric values
        model_names: list of model names in display order
        alpha: significance threshold (default 0.05)

    Returns:
        dict with:
          'matrix': (M, M) array of p-values (upper triangular)
          'significant': (M, M) bool array of FDR-corrected significance
          'stars': (M, M) string array for display
          'effect_sizes': (M, M) array of Cohen's d
          'n_pairs': number of paired observations per test
    """
    M = len(model_names)
    p_values = np.full((M, M), np.nan)
    effect_sizes = np.full((M, M), np.nan)

    # Collect all pairwise p-values for FDR correction
    all_pairs = []
    for i in range(M):
        for j in range(i + 1, M):
            a = per_day_values[model_names[i]]
            b = per_day_values[model_names[j]]

            # Ensure paired: must have same length
            assert len(a) == len(b), \
                f"Paired test requires equal-length arrays: {len(a)} vs {len(b)}"

            # Remove NaN pairs (e.g., WGAN-GP inf costs)
            mask = np.isfinite(a) & np.isfinite(b)
            if mask.sum() < 5:
                # Too few valid pairs — skip
                p_values[i, j] = np.nan
                continue

            a_valid, b_valid = a[mask], b[mask]

            # Wilcoxon signed-rank (paired)
            if np.allclose(a_valid, b_valid):
                p_values[i, j] = 1.0
                effect_sizes[i, j] = 0.0
            else:
                try:
                    stat, p = wilcoxon(a_valid, b_valid, alternative='two-sided')
                    p_values[i, j] = p
                except Exception:
                    p_values[i, j] = np.nan

                # Cohen's d for paired samples
                diff = a_valid - b_valid
                d = np.mean(diff) / max(np.std(diff), 1e-8)
                effect_sizes[i, j] = d

            all_pairs.append((i, j, p_values[i, j]))

    # Benjamini-Hochberg FDR correction
    valid_pairs = [(i, j, p) for i, j, p in all_pairs if not np.isnan(p)]
    n_tests = len(valid_pairs)
    valid_pairs.sort(key=lambda x: x[2])  # sort by p-value ascending

    fdr_thresholds = np.zeros(n_tests)
    for rank, (i, j, p) in enumerate(valid_pairs):
        fdr_thresholds[rank] = (rank + 1) / n_tests * alpha

    # Find the largest rank where p <= threshold
    significant_bh = np.full((M, M), False)
    for rank, (i, j, p) in enumerate(valid_pairs):
        if p <= fdr_thresholds[rank]:
            significant_bh[i, j] = True
        else:
            # All subsequent (higher rank) pairs are also not significant
            # (Benjamini-Hochberg procedure)
            pass

    # Re-apply: BH rejects all hypotheses where p <= max(p_i: p_i <= alpha*i/n)
    max_reject_rank = -1
    for rank, (i, j, p) in enumerate(valid_pairs):
        if p <= alpha * (rank + 1) / n_tests:
            max_reject_rank = rank

    significant_bh.fill(False)
    for rank, (i, j, p) in enumerate(valid_pairs):
        if rank <= max_reject_rank:
            significant_bh[i, j] = True

    # Star annotation
    stars = np.full((M, M), '', dtype=object)
    for i in range(M):
        for j in range(i + 1, M):
            p = p_values[i, j]
            if np.isnan(p):
                stars[i, j] = 'N/A'
            elif significant_bh[i, j]:
                if p < 0.001:
                    stars[i, j] = '***'
                elif p < 0.01:
                    stars[i, j] = '**'
                elif p < 0.05:
                    stars[i, j] = '*'
                else:
                    stars[i, j] = 'n.s.'
            else:
                stars[i, j] = 'n.s.'

    return {
        'matrix': p_values,
        'significant': significant_bh,
        'stars': stars,
        'effect_sizes': effect_sizes,
        'n_pairs': len(per_day_values[model_names[0]]) if model_names else 0,
    }


def extract_per_day_values(all_results, model_order, level_name, metric_key):
    """Extract per-day metric values from the flat evaluation results dict.

    Args:
        all_results: {(model_label, level_name, template_day): {metric_key: scalar}}
        model_order: list of model labels
        level_name: 'normal' | 'high_risk' | 'extreme'
        metric_key: e.g. 'crps_mean', 'tail_cal_error', 'daily_cost_mean'

    Returns:
        dict {model_label: (D,) array}, only for models that have data
    """
    # Find all unique template days for this (model, level) combination
    all_days = set()
    model_days = {}
    for (m_label, l_name, tday) in all_results:
        if l_name == level_name and m_label in model_order:
            all_days.add(tday)
            model_days.setdefault(m_label, set()).add(tday)

    common_days = sorted(all_days)

    result = {}
    for m in model_order:
        if m not in model_days:
            continue
        vals = []
        for d in common_days:
            key = (m, level_name, d)
            if key in all_results and metric_key in all_results[key]:
                v = all_results[key][metric_key]
                if isinstance(v, (int, float, np.floating, np.integer)):
                    vals.append(float(v))
                else:
                    vals.append(np.nan)
            else:
                vals.append(np.nan)
        if len(vals) >= 5 and not all(np.isnan(v) for v in vals):
            result[m] = np.array(vals)

    return result


def print_significance_matrix(stars, model_names, metric_label, level_label):
    """Print a formatted significance matrix.

    Args:
        stars: (M, M) string array from pairwise_wilcoxon()
        model_names: list of model display names
        metric_label: e.g. 'CRPS-mean'
        level_label: e.g. 'Normal Level'
    """
    M = len(model_names)
    short_names = [n.replace('\n', ' ') for n in model_names]

    # Max name width
    max_w = max(len(n) for n in short_names)

    print(f"\n{'=' * 130}")
    print(f"  Statistical Significance: {metric_label} — {level_label}")
    print(f"  Paired Wilcoxon signed-rank test, {M} models, FDR-corrected (Benjamini-Hochberg)")
    print(f"  *** p<0.001  ** p<0.01  * p<0.05  n.s. not significant")
    print(f"{'=' * 130}")

    # Header
    header = f"{'':>{max_w}s}"
    for n in short_names:
        header += f"  {n:>10s}"
    print(header)
    print('-' * 130)

    for i in range(M):
        row = f"{short_names[i]:>{max_w}s}"
        for j in range(M):
            if i == j:
                row += f"  {'—':>10s}"
            elif i < j:
                s = stars[i, j] if stars[i, j] else '?'
                row += f"  {s:>10s}"
            else:
                row += f"  {'':>10s}"  # lower triangle: empty
        print(row)

    print()


def compute_pairwise_effect_summary(per_day_values, model_names):
    """Compute per-model summary: best/worse count + mean rank.

    Returns a dict for printing a summary table.
    """
    M = len(model_names)
    # Build matrix of means
    means = np.array([np.nanmean(per_day_values[m]) for m in model_names])

    # For each metric direction, compute "better than" count
    # Lower is better for most metrics (CRPS, cost, etc.)
    better_count = np.zeros(M, dtype=int)
    for i in range(M):
        for j in range(M):
            if i != j and not np.isnan(means[i]) and not np.isnan(means[j]):
                if means[i] < means[j]:  # lower = better
                    better_count[i] += 1

    # Rank
    ranks = np.zeros(M)
    order = np.argsort(means)  # ascending = best first
    for rank, idx in enumerate(order):
        ranks[idx] = rank + 1

    return {
        'means': means,
        'better_than': better_count,
        'ranks': ranks,
    }


if __name__ == '__main__':
    # Quick self-test with synthetic data
    np.random.seed(42)
    model_names = ['V3 (DC-CAF)', 'V3-FM (DC-CAF-FM)', 'V3-IESGAT (DC-CAF-GAT)']
    n_days = 12

    per_day = {
        'V3 (DC-CAF)': np.random.normal(0.075, 0.005, n_days),
        'V3-FM (DC-CAF-FM)': np.random.normal(0.076, 0.007, n_days),
        'V3-IESGAT (DC-CAF-GAT)': np.random.normal(0.079, 0.008, n_days),
    }

    result = pairwise_wilcoxon(per_day, model_names, alpha=0.05)
    print_significance_matrix(
        result['stars'], model_names,
        metric_label='CRPS-mean', level_label='Normal Level'
    )

    # Show raw p-values
    M = len(model_names)
    print("Raw p-values (upper triangular):")
    for i in range(M):
        for j in range(i + 1, M):
            print(f"  {model_names[i]:>25s} vs {model_names[j]:<25s}: "
                  f"p={result['matrix'][i, j]:.4f}, "
                  f"d={result['effect_sizes'][i, j]:.2f}, "
                  f"sig={result['stars'][i, j]}")

    summary = compute_pairwise_effect_summary(per_day, model_names)
    print(f"\nRanks (lower=better): {summary['ranks']}")
