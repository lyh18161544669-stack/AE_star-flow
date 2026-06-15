"""Generate complete evaluation report — 12-day aggregation.

Reads _all12.csv per model (fallback to single-day CSV), evaluates all 3 levels,
and prints aggregated mean ± std across template days.
"""
import numpy as np, pandas as pd, os, sys, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, '.')

from sklearn.preprocessing import MinMaxScaler
from dataset_energy_v3 import load_energy_data, FEATURE_COLS
from evaluation_metrics import *
from ies_reliability_metrics import evaluate_physics_feasibility, compute_vss


def _resolve_csv(base_dir, model_name):
    all12 = f"{base_dir}/generated_scenarios_{model_name}_all12.csv"
    if os.path.exists(all12):
        return all12
    single = f"{base_dir}/generated_scenarios_{model_name}.csv"
    if os.path.exists(single):
        return single
    return None


MODEL_DIRS = [
    ('V1 (WGAN-GP)',           'results_energy_continuous_v1',           'v1'),
    ('V2 (ED-DDPM)',           'results_energy_continuous_v2',           'v2'),
    ('V2.5 (DC-DDPM)',         'results_energy_continuous_v2.5',         'v2.5'),
    ('V3 (DC-CAF)',            'results_energy_continuous_v3',           'v3'),
    ('V3-FM (DC-CAF-FM)',      'results_energy_continuous_v3_fm',        'v3_fm'),
    ('V3-IESGAT (DC-CAF-GAT)', 'results_energy_continuous_v3_iesgat',    'v3_iesgat'),
    ('V3-FM-IESGAT (DC-CAF-FM-GAT)','results_energy_continuous_v3_fm_iesgat','v3_fm_iesgat'),
    ('V4 (DC-GAT)',            'results_energy_continuous_v4',           'v4'),
]

LEVELS = {'normal': 0.0, 'high_risk': 0.90, 'extreme': 0.95}
MODEL_LABELS = ['WGAN-GP','ED-DDPM','DC-DDPM','DC-CAF','DC-CAF-FM','DC-CAF-GAT','DC-CAF-FM-GAT','DC-GAT']


def load_csv_scenarios(csv_path, scaler):
    """Load scenarios grouped by (level_name, template_day)."""
    if csv_path is None or not os.path.exists(csv_path):
        return {}
    df = pd.read_csv(csv_path, encoding='utf-8-sig')
    lvl_col = 'extreme_label' if 'extreme_label' in df.columns else 'extreme_level'
    td_col = 'template_day' if 'template_day' in df.columns else ('source_day' if 'source_day' in df.columns else None)

    results = {}
    for lvl_val in df[lvl_col].unique():
        lvl_name = str(lvl_val)
        if lvl_name not in LEVELS and lvl_name in ('v2_natural', 'v1_wgangp'):
            lvl_name = 'normal'
        if lvl_name not in LEVELS:
            continue

        subset = df[df[lvl_col] == lvl_val]
        tdays = subset[td_col].unique() if td_col else [181]
        for tday in tdays:
            if td_col:
                day_mask = subset[td_col] == tday
                day_vals = scaler.transform(subset[day_mask][FEATURE_COLS].values.astype(np.float32))
            else:
                day_vals = scaler.transform(subset[FEATURE_COLS].values.astype(np.float32))
            n_scenarios = len(day_vals) // 24
            if n_scenarios == 0:
                continue
            results[(lvl_name, int(tday))] = day_vals.reshape(n_scenarios, 24, 5)
    return results


# --- Load real data ---
raw, _ = load_energy_data('源荷数据集.csv')
full = raw.astype(np.float32)
scaler = MinMaxScaler(); scaler.fit(full)
norm = scaler.transform(full)
days_norm = norm.reshape(365, 24, 5)
days_nct = days_norm.transpose(0, 2, 1)
print(f"Real data: 365 days loaded")

# --- Evaluate all ---
all_results = {}  # {(label, lvl_name): {metric_key: [values_across_days]}}

for model_name, base_dir, model_key in MODEL_DIRS:
    csv_path = _resolve_csv(base_dir, model_key)
    label = model_key.replace('_', ' ').replace('v', 'V').replace('2.5', '2.5').replace('fm', 'FM').replace('iesgat', 'IESGAT').replace('IESGAT IESGAT', 'IESGAT')
    label = label.replace(' ', '-').replace('V-', 'V').upper()
    # Fix labels
    if label.startswith('V1'):
        label = 'WGAN-GP'
    elif label.startswith('V2.5'):
        label = 'DC-DDPM'
    elif label.startswith('V3-FM-IESGAT'):
        label = 'DC-CAF-FM-GAT'
    elif label.startswith('V3-FM'):
        label = 'DC-CAF-FM'
    elif label.startswith('V3-IESGAT'):
        label = 'DC-CAF-GAT'
    elif label.startswith('V3'):
        label = 'DC-CAF'
    elif label.startswith('V2'):
        label = 'ED-DDPM'
    elif label.startswith('V4'):
        label = 'DC-GAT'

    if csv_path is None:
        print(f"SKIP {label}: CSV not found")
        continue

    csv_data = load_csv_scenarios(csv_path, scaler)
    if not csv_data:
        print(f"SKIP {label}: empty")
        continue

    days_found = sorted(set(k[1] for k in csv_data.keys()))
    levels_found = sorted(set(k[0] for k in csv_data.keys()))
    print(f"\n{label}: {len(days_found)} days, levels={levels_found}, csv={os.path.basename(csv_path)}")

    for (lvl_name, tday), scenarios in sorted(csv_data.items()):
        n = scenarios.shape[0]
        gen_nct = scenarios.transpose(0, 2, 1)
        m = {}

        if lvl_name == 'normal':
            real_day = days_norm[tday].T
            crps = compute_crps_all(gen_nct, real_day)
            m['CRPS'] = crps['mean']
            m['CRPS-sum'] = crps['crps_sum']
            Xr = days_nct.reshape(365, -1)
            Xg = scenarios.reshape(n, -1)
            m['ED'] = energy_distance(Xr, Xg)
            m['SWD'] = sliced_wasserstein_distance(Xr, Xg, 200, 42)
            pit = compute_pit_histogram(gen_nct, real_day)
            m['PIT_chi2'] = pit['chi2_statistic']
            pr = compute_precision_recall_g(Xr, Xg, k=20)
            m['Prec_g'] = pr['precision']
            m['Rec_g'] = pr['recall']
            ci = compute_cr_iw(gen_nct, days_norm[tday])
            m['CR_95'] = ci['cr']
            m['IW_95'] = ci['iw']
            corr = compute_correlation_error(gen_nct, real_day)
            m['Corr'] = corr['mean']
            acf = compute_autocorr_error(gen_nct, real_day)
            m['ACF'] = acf['mean']
        else:
            tp = 90 if lvl_name == 'high_risk' else 95
            tc = compute_tail_calibration(gen_nct, days_nct, tp)
            m['TailCal'] = tc['mean']
            ec = compute_extreme_coverage_rate(gen_nct, days_nct, 95, lvl_name)
            m['ExtCov'] = ec['mean']
            ct = compute_crps_tail(gen_nct, days_nct, 90)
            m['CRPS-tail'] = ct['mean']
            Xr = days_nct.reshape(365, -1)
            Xg = gen_nct.reshape(n, -1)
            pr = compute_precision_recall_g(Xr, Xg, k=20)
            m['Rec_g'] = pr['recall']
            ci = compute_cr_iw(gen_nct, days_nct)
            m['CR_95'] = ci['cr']
            m['IW_95'] = ci['iw']

        phys = evaluate_physics_feasibility(scenarios, scaler=scaler)
        m['PFR'] = phys['feasible_rate']
        cost = phys['daily_operating_cost']
        m['Cost'] = cost['mean'] if cost else float('nan')
        m['Slack%'] = cost['slack_ratio'] if cost else float('nan')
        vss = compute_vss(scenarios, scaler=scaler)
        m['VSS'] = vss['vss']
        m['VSS%'] = vss['vss_relative']

        key = (label, lvl_name)
        if key not in all_results:
            all_results[key] = {k: [] for k in m}
        for k, v in m.items():
            if isinstance(v, (int, float, np.floating)):
                all_results[key][k].append(float(v))

# Aggregate across days
agg = {}
for key in all_results:
    agg[key] = {}
    for k, arr in all_results[key].items():
        agg[key][k] = {'mean': np.mean(arr), 'std': np.std(arr)} if arr else {'mean': float('nan'), 'std': float('nan')}


def print_sep(w=140):
    print('-' * w)


for lvl_name, lvl_display in [
    ('normal', 'TABLE 1: NORMAL (extreme_level=0.0) — Distribution Quality & Calibration (12d mean±std)'),
    ('high_risk', 'TABLE 2: HIGH-RISK (extreme_level=0.90) — Tail Consistency (12d mean±std)'),
    ('extreme', 'TABLE 3: EXTREME (extreme_level=0.95) — Tail Consistency (12d mean±std)')
]:
    models_in_level = [m for m in MODEL_LABELS if (m, lvl_name) in agg]
    if not models_in_level:
        continue

    n_days = max(len(all_results[(m, lvl_name)].get('CRPS', len(all_results[(m, lvl_name)].get('PFR', []))))
                 for m in models_in_level)
    print(f"\n{'='*150}")
    print(f"  {lvl_display}  [{n_days} test days]")
    print(f"  {len(models_in_level)} models")
    print(f"{'='*150}")

    if lvl_name == 'normal':
        metrics = [
            ('CRPS',       'CRPS',         'lower',  '.4f', False),
            ('CRPS-sum',   'CRPS-sum',     'lower',  '.4f', False),
            ('ED',         'Energy Dist',  'lower',  '.4f', False),
            ('SWD',        'SWD',          'lower',  '.4f', False),
            ('Prec_g',     'Precision_g',  'higher', '.4f', False),
            ('Rec_g',      'Recall_g',     'higher', '.4f', False),
            ('PIT_chi2',   'PIT chi2',     'lower',  '.1f', False),
            ('CR_95',      'CR_95',        '=95',    '.1f', True),
            ('IW_95',      'IW_95',        'lower',  '.4f', False),
            ('Corr',       'Corr Err',     'lower',  '.4f', False),
            ('ACF',        'ACF Err',      'lower',  '.4f', False),
            ('PFR',        'LP Feasible',  'higher', '.1f', True),
            ('Slack%',     'Slack Ratio',  'lower',  '.1f', True),
            ('VSS',        'VSS (yuan)',   'higher', '.0f', False),
            ('VSS%',       'VSS (%)',      'higher', '.2f', True),
        ]
    else:
        metrics = [
            ('TailCal',    'Tail Cal Err', 'lower',  '.4f', False),
            ('ExtCov',     'Ext Coverage', 'higher', '.1f', False),
            ('CRPS-tail',  'CRPS-tail',    'lower',  '.4f', False),
            ('Rec_g',      'Recall_g',     'higher', '.4f', False),
            ('CR_95',      'CR_95',        '=95',    '.1f', True),
            ('IW_95',      'IW_95',        'lower',  '.4f', False),
            ('PFR',        'LP Feasible',  'higher', '.1f', True),
            ('Slack%',     'Slack Ratio',  'lower',  '.1f', True),
            ('VSS',        'VSS (yuan)',   'higher', '.0f', False),
            ('VSS%',       'VSS (%)',      'higher', '.2f', True),
        ]

    # Best values
    best = {}
    for key, name, direction, fmt, is_pct in metrics:
        vals = [(m, agg[(m, lvl_name)][key]['mean']) for m in models_in_level
                if key in agg[(m, lvl_name)] and not np.isnan(agg[(m, lvl_name)][key]['mean'])]
        if not vals:
            continue
        if direction == 'lower':
            best[key] = min(vals, key=lambda x: x[1])
        elif direction == 'higher':
            best[key] = max(vals, key=lambda x: x[1])
        elif direction == '=95':
            best[key] = min(vals, key=lambda x: abs(x[1] - 0.95))

    # Header
    hdr = f"  {'Metric':<18s}"
    for m in models_in_level:
        hdr += f"  {m:>14s}"
    print(hdr)
    print_sep(150)

    for key, name, direction, fmt, is_pct in metrics:
        row = f"  {name:<18s}"
        bm = best.get(key, (None,))[0]
        for m in models_in_level:
            entry = agg[(m, lvl_name)].get(key)
            if entry is None or np.isnan(entry['mean']):
                row += f"  {'-':>14s}"
                continue
            v, s = entry['mean'], entry['std']
            if key == 'CR_95':
                s_str = f"{v*100:7.1f}±{s*100:.1f}%"
            elif is_pct:
                s_str = f"{v:7.1f}±{s:.1f}%"
            elif key == 'ExtCov':
                s_str = f"{v:7.2f}±{s:.2f}x"
            elif 'VSS' in key or key == 'Rec_g' or key == 'Prec_g':
                s_str = f"{v:9.4f}±{s:.4f}"
            elif 'Cost' in key:
                s_str = f"{v:9.0f}±{s:.0f}"
            else:
                s_str = f"{v:9.4f}±{s:.4f}"
            s_str += ' *' if m == bm else '  '
            row += f"  {s_str:>14s}"
        print(row)

    print(f"\n  * = best in metric")

print(f"\n{'='*150}")
print("  NOTES:")
print("  - CRPS/ED/SWD: lower=better. PIT chi2: lower=better (uniform=0).")
print("  - Tail Cal Err: lower=better calibrated. Ext Coverage: higher=more extreme coverage.")
print("  - CR_95 ideal=95%. LP Feasible: higher=better.")
print("  - VSS = EEV - WS via deviation penalty. Positive = stochastic adds value.")
print("  - 12-day mean ± std reflects seasonal robustness.")
print(f"{'='*150}")
