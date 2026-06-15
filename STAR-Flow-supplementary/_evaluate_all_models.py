"""
Comprehensive model evaluation with 3-level hierarchical metrics.

Level system:
  Normal    (0.0)  → CRPS, ED, SWD, PIT, precision/recall, CR/IW, LP cost
  High-Risk (0.90) → Tail Cal., Ext. Coverage, CR/IW, recall, LP cost
  Extreme   (0.95) → Tail Cal., Ext. Coverage, CR/IW, recall, LP cost

Key fix: CRPS is only computed for Normal level against the template_day's
real observation. Extreme levels use tail calibration (does the generated
P95 median match real data P95?) instead of CRPS (which would penalize
the model for successfully deviating from normal observations).

Refs: CDSG (Zhao et al. 2025, Applied Energy), TailDiff (Naumov et al. 2024)
"""

import numpy as np
import pandas as pd
import os, sys, warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import MinMaxScaler

from evaluation_metrics import (
    compute_crps_all, energy_distance, sliced_wasserstein_distance,
    compute_moment_errors, compute_correlation_error, compute_autocorr_error,
    compute_pit_histogram, compute_precision_recall_g, compute_cr_iw,
    compute_tail_calibration, compute_extreme_coverage_rate, compute_crps_tail,
)
from ies_reliability_metrics import evaluate_physics_feasibility, compute_vss
from _statistical_tests import (
    pairwise_wilcoxon, extract_per_day_values,
    print_significance_matrix, compute_pairwise_effect_summary,
)

DATA_PATH = "./源荷数据集.csv"
FEATURE_COLS = ['wind', 'solar', 'electric', 'heat', 'cold']

# ---- 3-Level System ----
LEVEL_CONFIG = {
    'normal':    {'extreme_level': 0.0,  'display': 'Normal (0.0)',
                  'metric_mode': 'calibration'},
    'high_risk': {'extreme_level': 0.90, 'display': 'High-Risk (0.90)',
                  'metric_mode': 'tail'},
    'extreme':   {'extreme_level': 0.95, 'display': 'Extreme (0.95)',
                  'metric_mode': 'tail'},
}

# Map legacy CSV extreme_level values to new level names
LEVEL_VALUE_MAP = {0.0: 'normal', 0.5: None, 0.90: 'high_risk',
                   0.95: 'extreme', 0.99: None}

def _resolve_csv(base_dir, model_name):
    """Resolve CSV path: prefer _all12.csv, fall back to single-day CSV."""
    all12 = f"{base_dir}/generated_scenarios_{model_name}_all12.csv"
    if os.path.exists(all12):
        return all12
    single = f"{base_dir}/generated_scenarios_{model_name}.csv"
    if os.path.exists(single):
        return single
    return None


MODEL_CONFIGS = {
    'ConvCVAE\n(CNN-VAE)': {
        'dir': 'results_cvae_baseline',
        'csv': 'results_cvae_baseline/generated_scenarios_cvae_all12.csv',
        'label': 'ConvCVAE',
        'model_key': 'cvae',
    },
    'V1\n(WGAN-GP)': {
        'dir': 'results_energy_continuous_v1',
        'csv': 'results_energy_continuous_v1/generated_scenarios_v1.csv',
        'label': 'WGAN-GP',
        'model_key': 'v1',
    },
    'V2\n(ED-DDPM)': {
        'dir': 'results_energy_continuous_v2',
        'csv': 'results_energy_continuous_v2/generated_scenarios_v2.csv',
        'label': 'ED-DDPM',
        'model_key': 'v2',
    },
    'V2.5\n(DC-DDPM)': {
        'dir': 'results_energy_continuous_v2.5',
        'csv': 'results_energy_continuous_v2.5/generated_scenarios_v2.5.csv',
        'label': 'DC-DDPM',
        'model_key': 'v2.5',
    },
    'V3\n(DC-CAF)': {
        'dir': 'results_energy_continuous_v3',
        'csv': 'results_energy_continuous_v3/generated_scenarios_v3.csv',
        'label': 'DC-CAF',
        'model_key': 'v3',
    },
    'V3-FM\n(DC-CAF-FM)': {
        'dir': 'results_energy_continuous_v3_fm',
        'csv': 'results_energy_continuous_v3_fm/generated_scenarios_v3_fm.csv',
        'label': 'DC-CAF-FM',
        'model_key': 'v3_fm',
    },
    'V3-IESGAT\n(DC-CAF-GAT)': {
        'dir': 'results_energy_continuous_v3_iesgat',
        'csv': 'results_energy_continuous_v3_iesgat/generated_scenarios_v3_iesgat.csv',
        'label': 'DC-CAF-GAT',
        'model_key': 'v3_iesgat',
    },
    'V3-FM-IESGAT\n(DC-CAF-FM-GAT)': {
        'dir': 'results_energy_continuous_v3_fm_iesgat',
        'csv': 'results_energy_continuous_v3_fm_iesgat/generated_scenarios_v3_fm_iesgat.csv',
        'label': 'DC-CAF-FM-GAT',
        'model_key': 'v3_fm_iesgat',
    },
    'V4\n(DC-GAT)': {
        'dir': 'results_energy_continuous_v4',
        'csv': 'results_energy_continuous_v4/generated_scenarios_v4.csv',
        'label': 'DC-GAT',
        'model_key': 'v4',
    },
}

LEGACY_LABEL_MAP = {'v1_wgangp': 'normal', 'v2_natural': 'normal'}


# ===========================================================================
# Data Loading
# ===========================================================================

def load_real_data():
    from dataset_energy_v3 import load_energy_data
    raw, _ = load_energy_data(DATA_PATH)
    full = raw.astype(np.float32)
    scaler = MinMaxScaler()
    scaler.fit(full)
    norm = scaler.transform(full)
    days_norm = norm.reshape(365, 24, 5)
    days_raw = full.reshape(365, 24, 5)
    return days_norm, days_raw, scaler


def get_test_days():
    """Get test day indices (1st of each month, 12 days)."""
    from dataset_energy_v3 import get_monthly_stratified_split
    _, _, test_days = get_monthly_stratified_split(DATA_PATH)
    return test_days


# ===========================================================================
# CSV Loading — tracks template_day for conditional CRPS
# ===========================================================================

def load_csv_scenarios(csv_path, scaler):
    """Load scenarios grouped by (level_name, template_day).

    Returns:
        {(level_name, template_day): (N, 24, 5) array}
        level_name ∈ {'normal', 'high_risk', 'extreme'}

    Legacy levels 0.5 (moderate) and 0.99 (severe) are skipped.
    """
    if csv_path is None or not os.path.exists(csv_path):
        return {}

    df = pd.read_csv(csv_path, encoding='utf-8-sig')

    # Detect level column
    if 'extreme_label' in df.columns:
        level_col = 'extreme_label'
    else:
        level_col = 'extreme_level'

    template_day_col = 'template_day' if 'template_day' in df.columns else None

    if level_col == 'extreme_label':
        levels = df[level_col].unique()
    else:
        levels = sorted(df[level_col].unique())

    results = {}
    for level_val in levels:
        # Map to new 3-level system
        if level_col == 'extreme_label':
            level_name = LEGACY_LABEL_MAP.get(str(level_val), str(level_val))
        else:
            fval = float(level_val)
            level_name = LEVEL_VALUE_MAP.get(fval, str(level_val))

        # Skip levels not in the new 3-level system
        if level_name is None or level_name not in LEVEL_CONFIG:
            continue

        subset = df[df[level_col] == level_val]
        raw_vals = subset[FEATURE_COLS].values.astype(np.float32)
        norm_vals = scaler.transform(raw_vals)

        # Group by template_day
        if template_day_col and template_day_col in df.columns:
            tdays = subset[template_day_col].unique()
        else:
            tdays = [181]  # default for legacy data

        for tday in tdays:
            if template_day_col and template_day_col in df.columns:
                day_mask = subset[template_day_col] == tday
                day_vals = scaler.transform(
                    subset[day_mask][FEATURE_COLS].values.astype(np.float32)
                )
            else:
                day_vals = norm_vals

            n_scenarios = len(day_vals) // 24
            if n_scenarios == 0:
                continue
            arr = day_vals.reshape(n_scenarios, 24, 5)
            key = (level_name, int(tday))
            results[key] = arr

    return results


# ===========================================================================
# Level-Aware Evaluation
# ===========================================================================

def evaluate_normal_level(scenarios, real_days_norm, real_days_nct, days_raw,
                           scaler, template_day):
    """Normal level: full calibration metrics.

    CRPS is computed against the template_day's real observation.
    ED/SWD are computed against the full 365-day distribution.

    Shape convention:
      scenarios:     (N, T, C) = (N, 24, 5)
      real_days_nct: (D, C, T) = (365, 5, 24)
    """
    N = scenarios.shape[0]
    gen_nct = scenarios.transpose(0, 2, 1)          # (N, C, T) = (N, 5, 24)
    real_day_ct = real_days_nct[template_day]        # (C, T) = (5, 24)
    real_day_tc = real_days_norm[template_day]       # (T, C) = (24, 5)

    m = {}

    # CRPS — conditional: generated[day=d] vs real[day=d]
    crps = compute_crps_all(gen_nct, real_day_ct)
    m['crps_mean'] = crps['mean']
    m['crps_sum'] = crps['crps_sum']
    m['crps_per_var'] = crps['per_variable']

    # Energy Distance & SWD vs full 365-day distribution
    # Use consistent flatten order (C, T) matching real_days_nct
    X_real = real_days_nct.reshape(365, -1)
    X_gen = gen_nct.reshape(N, -1)
    m['energy_distance'] = energy_distance(X_real, X_gen)
    m['swd'] = sliced_wasserstein_distance(X_real, X_gen, n_projections=200, seed=42)

    # PIT histogram — calibration diagnostic
    pit = compute_pit_histogram(gen_nct, real_day_ct)
    m['pit_chi2'] = pit['chi2_statistic']
    m['pit_ks'] = pit['ks_statistic']

    # precision / recall
    pr = compute_precision_recall_g(real_days_nct.reshape(365, -1), X_gen, k=20)
    m['precision_g'] = pr['precision']
    m['recall_g'] = pr['recall']

    # Moment errors
    moments = compute_moment_errors(gen_nct, real_days_nct)
    m['mean_abs_err'] = moments['avg']['mean_err']
    m['std_abs_err'] = moments['avg']['std_err']

    # Correlation & ACF (against template_day)
    corr = compute_correlation_error(gen_nct, real_day_ct)
    m['corr_frobenius'] = corr['mean']
    acf = compute_autocorr_error(gen_nct, real_day_ct, max_lag=6)
    m['acf_mean_err'] = acf['mean']

    # CR / IW (against template_day)
    cr_iw = compute_cr_iw(gen_nct, real_day_tc)
    m['cr_95'] = cr_iw['cr']
    m['iw_95'] = cr_iw['iw']

    # LP dispatch feasibility
    phys = evaluate_physics_feasibility(scenarios, scaler=scaler)
    m['lp_feasible_rate'] = phys['feasible_rate']
    m['lp_feasible_count'] = phys['feasible_count']
    m['lp_total'] = phys['total_scenarios']
    cost = phys['daily_operating_cost']
    if cost is not None:
        m['daily_cost_mean'] = cost['mean']
        m['daily_cost_std'] = cost['std']
        m['daily_slack_mean'] = cost['mean_slack']
        m['slack_ratio'] = cost['slack_ratio']

    # VSS: does stochastic dispatch beat deterministic?
    vss = compute_vss(scenarios, scaler=scaler)
    m['vss'] = vss['vss']
    m['vss_relative'] = vss['vss_relative']
    m['ws_cost'] = vss['ws_cost']
    m['eev_cost'] = vss['eev_cost']

    return m


def evaluate_tail_level(scenarios, real_days_norm, real_days_nct, days_raw,
                         scaler, level_name, template_day):
    """High-Risk / Extreme level: tail consistency metrics.

    CRPS is NOT computed — it would penalize the model for successfully
    generating scenarios that deviate from normal observations.
    Instead: tail calibration + extreme coverage + CRPS-tail + CR/IW + LP.

    Shape convention:
      scenarios:     (N, T, C) = (N, 24, 5)
      real_days_nct: (D, C, T) = (365, 5, 24)
    """
    N = scenarios.shape[0]
    gen_nct = scenarios.transpose(0, 2, 1)  # (N, C, T)

    target_pct = 90 if level_name == 'high_risk' else 95

    m = {}

    # Tail calibration: does gen median match real P90/P95?
    tc = compute_tail_calibration(gen_nct, real_days_nct, target_percentile=target_pct)
    m['tail_cal_error'] = tc['mean']
    m['tail_cal_per_var'] = tc['per_variable']

    # Extreme coverage: does gen cover the right fraction of real extremes?
    ec = compute_extreme_coverage_rate(gen_nct, real_days_nct, percentile=target_pct,
                                        level_name=level_name)
    m['ext_coverage_rate'] = ec['mean']
    m['ext_coverage_per_var'] = ec['per_variable']

    # CRPS-tail: distribution distance in tail region only
    ct = compute_crps_tail(gen_nct, real_days_nct, tail_pct=90)
    m['crps_tail'] = ct['mean']

    # CR / IW (against all real days for coverage assessment)
    cr_iw = compute_cr_iw(gen_nct, real_days_nct)
    m['cr_95'] = cr_iw['cr']
    m['iw_95'] = cr_iw['iw']

    # precision/recall — recall is key for extreme diversity
    X_real_ct = real_days_nct.reshape(365, -1)
    X_gen_ct = gen_nct.reshape(N, -1)  # same (C, T) order as real_days_nct
    pr = compute_precision_recall_g(X_real_ct, X_gen_ct, k=20)
    m['precision_g'] = pr['precision']
    m['recall_g'] = pr['recall']

    # LP dispatch feasibility
    phys = evaluate_physics_feasibility(scenarios, scaler=scaler)
    m['lp_feasible_rate'] = phys['feasible_rate']
    m['lp_feasible_count'] = phys['feasible_count']
    m['lp_total'] = phys['total_scenarios']
    cost = phys['daily_operating_cost']
    if cost is not None:
        m['daily_cost_mean'] = cost['mean']
        m['daily_cost_std'] = cost['std']
        m['daily_slack_mean'] = cost['mean_slack']
        m['slack_ratio'] = cost['slack_ratio']

    # VSS: does stochastic dispatch beat deterministic?
    vss = compute_vss(scenarios, scaler=scaler)
    m['vss'] = vss['vss']
    m['vss_relative'] = vss['vss_relative']

    return m


def evaluate_model_level(scenarios, real_days_norm, real_days_nct, days_raw,
                          scaler, level_name, template_day):
    """Dispatcher: routes to calibration or tail evaluation based on level."""
    mode = LEVEL_CONFIG[level_name]['metric_mode']
    if mode == 'calibration':
        return evaluate_normal_level(scenarios, real_days_norm, real_days_nct,
                                      days_raw, scaler, template_day)
    else:
        return evaluate_tail_level(scenarios, real_days_norm, real_days_nct,
                                    days_raw, scaler, level_name, template_day)


# ===========================================================================
# Print Helpers
# ===========================================================================

def sep(width=120):
    print("-" * width)

def header(text, width=120):
    print(f"\n{'=' * width}")
    print(f"  {text}")
    print(f"{'=' * width}")


# ===========================================================================
# Main
# ===========================================================================

def main():
    header("3-Level Hierarchical Model Evaluation", 130)
    print("Levels: Normal (calibration) | High-Risk P90 (tail) | Extreme P95 (tail)")
    print("Key: CRPS only for Normal. Tail calibration for High-Risk/Extreme.")

    # Load real data
    print("\nLoading real data...")
    days_norm, days_raw, scaler = load_real_data()
    days_nct = days_norm.transpose(0, 2, 1)   # (D, T, C) → (D, C, T) for metric fns
    test_days = get_test_days()
    print(f"  Real data: {len(days_norm)} days, {len(test_days)} test days")

    # Real data baselines
    print("\nComputing real data baselines...")
    real_phys = evaluate_physics_feasibility(days_norm, scaler=scaler)
    real_cost = real_phys['daily_operating_cost']
    print(f"  LP feasible: {real_phys['feasible_rate']:.1f}% "
          f"({real_phys['feasible_count']}/{real_phys['total_scenarios']})")
    if real_cost:
        print(f"  Daily cost: {real_cost['mean']:.0f} ± {real_cost['std']:.0f} yuan")
        print(f"  Slack ratio: {real_cost['slack_ratio']:.1f}%")

    # ---- Evaluate All Models ----
    all_results = {}  # {(model_label, level_name, template_day): metrics}

    for model_name, config in MODEL_CONFIGS.items():
        model_label = config['label']

        # Prefer _all12.csv, fall back to single-day CSV
        csv_path = _resolve_csv(config['dir'], config['model_key'])
        if csv_path is None:
            print(f"\n  WARNING: No CSV found for {model_label}")
            continue

        csv_label = csv_path.replace('\\', '/')
        header(f"Evaluating: {model_label}  [{csv_label.split('/')[-1]}]", 130)

        csv_data = load_csv_scenarios(csv_path, scaler)

        if not csv_data:
            print(f"  WARNING: No CSV data found for {model_label}")
            continue

        # Report what was found
        levels_found = sorted(set(k[0] for k in csv_data.keys()))
        days_found = sorted(set(k[1] for k in csv_data.keys()))
        print(f"  Levels found: {levels_found}")
        print(f"  Template days ({len(days_found)}): {days_found}")

        for (level_name, tday), scenarios in sorted(csv_data.items()):
            n_scenarios = scenarios.shape[0]
            print(f"  [{level_name}] day={tday}: {n_scenarios} scenarios")

            metrics = evaluate_model_level(
                scenarios, days_norm, days_nct, days_raw, scaler,
                level_name, tday
            )
            all_results[(model_label, level_name, tday)] = metrics

    # ---- Summary Tables ----
    header("RESULTS SUMMARY", 140)

    model_order = ['ConvCVAE', 'WGAN-GP', 'ED-DDPM', 'DC-DDPM', 'DC-CAF',
                   'DC-CAF-FM', 'DC-CAF-GAT', 'DC-CAF-FM-GAT', 'DC-GAT']

    # Aggregate: for each (model, level), average across template days
    # agg stores {mean, std} for each metric
    agg = {}  # {(model_label, level_name): {metric_key: {'mean': ..., 'std': ...}}}
    for (m_label, l_name, _), metrics in all_results.items():
        key = (m_label, l_name)
        if key not in agg:
            agg[key] = {k: [] for k in metrics.keys()}
        for k, v in metrics.items():
            if isinstance(v, (int, float, np.floating, np.integer)):
                agg[key][k].append(float(v))

    n_days_found = {}  # {(model_label, level_name): num_days}
    for key in agg:
        vals_by_key = list(agg[key].values())
        n_days = max((len(v) for v in vals_by_key), default=0)
        n_days_found[key] = n_days
        for k in agg[key]:
            arr = agg[key][k]
            agg[key][k] = {'mean': np.mean(arr), 'std': np.std(arr)} if arr else {
                'mean': float('nan'), 'std': float('nan')}

    # ---- TABLE 1: Normal Level Distribution Quality ----
    print(f"\n{'=' * 140}")
    print("  TABLE 1: Distribution Quality — Normal Level (12-day mean ± std)")
    print("  CRPS is conditional: generated[day=d] vs real[day=d]. Lower = better.")
    print(f"{'=' * 140}")

    dist_metrics = [
        ('crps_mean', 'CRPS (mean)', '↓'),
        ('crps_sum', 'CRPS-sum', '↓'),
        ('energy_distance', 'Energy Dist.', '↓'),
        ('swd', 'SWD', '↓'),
        ('precision_g', 'Precision_g', '↑'),
        ('recall_g', 'Recall_g', '↑'),
        ('pit_chi2', 'PIT chi2', '↓'),
        ('cr_95', 'CR_95 (%)', '≈95'),
        ('iw_95', 'IW_95', '↓'),
        ('corr_frobenius', 'Corr Err (Fro)', '↓'),
        ('acf_mean_err', 'ACF Err', '↓'),
        ('vss', 'VSS (yuan)', '↑'),
        ('vss_relative', 'VSS (%)', '↑'),
    ]

    # Filter to only models that have normal data
    normal_models = [m for m in model_order if (m, 'normal') in agg]
    n_days_info = f"  {n_days_found.get((normal_models[0], 'normal'), '?')}d" if normal_models else ""
    print(f"{'Metric':>28s}" + "".join(f"  {m:>15s}" for m in normal_models))
    sep(140)

    for metric_key, metric_label, direction in dist_metrics:
        row = f"{metric_label:>28s}"
        for m in normal_models:
            entry = agg[(m, 'normal')].get(metric_key, None)
            if entry is None or np.isnan(entry['mean']):
                row += f"  {'-':>15s}"
                continue
            v, s = entry['mean'], entry['std']
            if 'rate' in metric_key or metric_key == 'cr_95':
                row += f"  {v*100:9.1f}±{s*100:.1f}"
            else:
                row += f"  {v:9.4f}±{s:.4f}"
        print(row)

    # ---- TABLE 2: Tail Calibration (High-Risk & Extreme) ----
    for lvl_name, lvl_display in [('extreme', 'Extreme (P95)'),
                                     ('high_risk', 'High-Risk (P90)')]:
        print(f"\n{'=' * 140}")
        print(f"  TABLE 2{chr(ord('a') + (0 if lvl_name == 'extreme' else 1))}: "
              f"Tail Calibration — {lvl_display} Level (12-day mean ± std)")
        print(f"  CRPS is NOT used here (would penalize successful deviation).")
        print(f"  Tail Cal. Error: |gen_median - real_Pxx| / real_Pxx. Lower = better calibrated.")
        print(f"  Ext. Coverage: gen_extreme_rate / ideal_rate. 1.0x = perfect. <1 = under-generates extremes.")
        print(f"{'=' * 140}")

        tail_metrics = [
            ('tail_cal_error', 'Tail Cal. Error', '↓'),
            ('ext_coverage_rate', 'Ext. Coverage', '≈1.0'),
            ('crps_tail', 'CRPS-tail (WD)', '↓'),
            ('recall_g', 'Recall_g', '↑'),
            ('cr_95', 'CR_95 (%)', '≈95'),
            ('iw_95', 'IW_95', '↓'),
            ('lp_feasible_rate', 'LP Feasible (%)', '↑'),
            ('daily_cost_mean', 'Daily Cost (yuan)', '↓'),
            ('slack_ratio', 'Slack Ratio (%)', '↓'),
            ('vss', 'VSS (yuan)', '↑'),
            ('vss_relative', 'VSS (%)', '↑'),
        ]

        tail_models = [m for m in model_order if (m, lvl_name) in agg]
        if not tail_models:
            print(f"  (No {lvl_name} data available — re-sample with extreme_level={LEVEL_CONFIG[lvl_name]['extreme_level']})")
            continue

        print(f"{'Metric':>28s}" + "".join(f"  {m:>15s}" for m in tail_models))
        sep(140)

        for metric_key, metric_label, direction in tail_metrics:
            row = f"{metric_label:>28s}"
            for m in tail_models:
                entry = agg[(m, lvl_name)].get(metric_key, None)
                if entry is None or np.isnan(entry['mean']):
                    row += f"  {'-':>15s}"
                    continue
                v, s = entry['mean'], entry['std']
                if metric_key == 'ext_coverage_rate':
                    # Ratio: 1.0 = ideal, >1 = over-generate, <1 = under-generate
                    row += f"  {v:9.2f}±{s:.2f}x"
                elif 'rate' in metric_key or metric_key == 'cr_95' or 'feasible' in metric_key:
                    if v <= 1.0:
                        row += f"  {v*100:9.1f}±{s*100:.1f}"
                    else:
                        row += f"  {v:9.1f}±{s:.1f}"
                elif 'cost' in metric_key:
                    row += f"  {v:10.0f}±{s:.0f}"
                else:
                    row += f"  {v:9.4f}±{s:.4f}"
            print(row)

    # ---- TABLE 3: LP Dispatch Cost (All Levels) ----
    print(f"\n{'=' * 140}")
    print("  TABLE 3: LP Dispatch Feasibility & Cost — All Levels (12-day mean ± std)")
    print(f"  Real data baseline: PFR={real_phys['feasible_rate']:.1f}%, "
          f"Cost={real_cost['mean']:.0f} yuan, Slack={real_cost['slack_ratio']:.1f}%")
    print(f"{'=' * 140}")

    lp_metrics = [
        ('lp_feasible_rate', 'LP Feasible (%)', '↑'),
        ('daily_cost_mean', 'Daily Cost (yuan)', '↓'),
        ('slack_ratio', 'Slack Ratio (%)', '↓'),
    ]

    for lvl_name in ['normal', 'high_risk', 'extreme']:
        lp_models = [m for m in model_order if (m, lvl_name) in agg]
        if not lp_models:
            continue
        lvl_display = LEVEL_CONFIG[lvl_name]['display']
        print(f"\n  [{lvl_display}]")
        print(f"  {'Metric':>28s}" + "".join(f"  {m:>15s}" for m in lp_models))
        sep(140)
        for metric_key, metric_label, direction in lp_metrics:
            row = f"  {metric_label:>26s}"
            for m in lp_models:
                entry = agg[(m, lvl_name)].get(metric_key, None)
                if entry is None or np.isnan(entry['mean']):
                    row += f"  {'-':>15s}"
                    continue
                v, s = entry['mean'], entry['std']
                if 'rate' in metric_key or 'feasible' in metric_key:
                    row += f"  {v:9.1f}±{s:.1f}"
                elif 'cost' in metric_key:
                    row += f"  {v:10.0f}±{s:.0f}"
                else:
                    row += f"  {v:9.1f}±{s:.1f}"
            print(row)

    # ---- TABLE 4: Statistical Significance Tests ----
    print(f"\n{'=' * 130}")
    print("  TABLE 4: Statistical Significance (Paired Wilcoxon, FDR-corrected)")
    print("  Unit of analysis: 12 test days. NOT scenarios (avoids pseudo-replication).")
    print(f"{'=' * 130}")

    # Helper to run significance for a (metric, level) combo
    def run_sig_test(metric_key, level_name, metric_label):
        per_day = extract_per_day_values(all_results, model_order, level_name, metric_key)
        if len(per_day) < 2:
            print(f"\n  [{metric_label} / {level_name}]: Not enough models for comparison")
            return
        result = pairwise_wilcoxon(per_day, model_order, alpha=0.05)
        print_significance_matrix(
            result['stars'], model_order,
            metric_label=metric_label, level_label=f'{level_name} Level'
        )

    # Key metrics for significance testing
    sig_tests = [
        # (metric_key, level_name, display_label)
        ('crps_mean', 'normal', 'CRPS-mean (Normal)'),
        ('crps_sum', 'normal', 'CRPS-sum (Normal)'),
        ('energy_distance', 'normal', 'Energy Distance (Normal)'),
        ('pit_chi2', 'normal', 'PIT chi2 (Normal)'),
        ('tail_cal_error', 'extreme', 'Tail Cal Error (Extreme P95)'),
        ('ext_coverage_rate', 'extreme', 'Ext Coverage (Extreme P95)'),
        ('daily_cost_mean', 'extreme', 'Daily Cost (Extreme P95)'),
        ('lp_feasible_rate', 'extreme', 'LP Feasible Rate (Extreme P95)'),
        ('tail_cal_error', 'high_risk', 'Tail Cal Error (High-Risk P90)'),
    ]

    for metric_key, level_name, metric_label in sig_tests:
        run_sig_test(metric_key, level_name, metric_label)

    print(f"\n{'=' * 130}")
    print("  Evaluation Complete!")
    print(f"{'=' * 130}")


if __name__ == '__main__':
    main()
