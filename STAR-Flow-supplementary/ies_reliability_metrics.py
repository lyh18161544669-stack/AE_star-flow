"""
IES Reliability Metrics — IEEE Std 1366 / IEC 61000 compliance.

Core metrics:
  1. LOLP  (Loss of Load Probability)  — Pr(shed > 1 kW) per IEEE 1366
  2. EENS  (Expected Energy Not Served) — kWh/day absolute + % of total load
  3. VSS   (Value of Stochastic Solution) — proper 2-stage w/ re-dispatch
  4. Slack Duration Curve data

Key distinctions from physics_feasibility_metrics.py:
  - RELIABILITY (LOLP/EENS) vs. FEASIBILITY (PFR binary)
  - Continuous severity: "how much load is shed?" vs. "feasible or not?"
  - IEEE 1366 bus-level bottleneck identification
  - Proper 2-stage VSS (re-dispatch with fixed GT) — not penalty-based approximation
  - Slack duration curves for visual comparison

Refs:
  IEEE Std 1366-2012 (distribution reliability indices)
  CDSG (Zhao et al. 2025, Applied Energy)
  AugDiT (Xie et al. 2025, Applied Energy)
  Birge & Louveaux, "Introduction to Stochastic Programming" (2011)
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os, json, warnings
warnings.filterwarnings('ignore')

# ===========================================================================
# IEEE 1366 Reliability Metrics
# ===========================================================================

def compute_lolp(lp_results, T=24, threshold_kw=1.0):
    """LOLP: Loss of Load Probability per IEEE Std 1366.

    Definition:
      LOLP = Pr(shed_load > ε) over all scenario-hours
           = (# hours with any shedding > threshold) / total_hours

    IEEE 1366 defines LOLE (hours/year) and ENS (Energy Not Supplied).
    LOLP is the fractional form: LOLP * 8760 = LOLE (h/yr).

    Bus-level breakdown identifies which energy bus (elec/heat/cold) is
    the bottleneck — critical for IES multi-energy systems.

    Args:
        lp_results: list of result dicts from IESOptimizer.solve() / solve_with_fixed_gt()
        T: periods per day (24)
        threshold_kw: kW threshold for counting a shed event (default 1.0 kW)

    Returns:
        dict with: lolp, lolp_elec, lolp_heat, lolp_cold,
                   lo_le_h_per_year, n_scenarios, total_hours
    """
    n = len(lp_results)
    total_hours = n * T

    shed_by_hour = np.zeros(total_hours)
    shed_elec_by_hour = np.zeros(total_hours)
    shed_heat_by_hour = np.zeros(total_hours)
    shed_cold_by_hour = np.zeros(total_hours)

    idx = 0
    for r in lp_results:
        if not r.get('feasible', False):
            continue
        if 'per_hour' not in r:
            continue
        ph = r['per_hour']
        shed_by_hour[idx:idx+T]      = ph['shed']
        shed_elec_by_hour[idx:idx+T] = ph['shed_elec']
        shed_heat_by_hour[idx:idx+T] = ph['shed_heat']
        shed_cold_by_hour[idx:idx+T] = ph['shed_cold']
        idx += T

    valid_hours = idx
    if valid_hours == 0:
        return {
            'lolp': 0.0, 'lolp_elec': 0.0, 'lolp_heat': 0.0, 'lolp_cold': 0.0,
            'lo_le_h_per_year': 0.0,
            'n_scenarios': n, 'total_hours': 0,
        }

    n_scenarios = valid_hours // T

    # Fraction of hours with shedding > threshold
    lolp       = float((shed_by_hour[:valid_hours]       > threshold_kw).mean())
    lolp_elec  = float((shed_elec_by_hour[:valid_hours]  > threshold_kw).mean())
    lolp_heat  = float((shed_heat_by_hour[:valid_hours]  > threshold_kw).mean())
    lolp_cold  = float((shed_cold_by_hour[:valid_hours]  > threshold_kw).mean())

    # LOLE = LOLP * 8760 h/yr (IEEE 1366 annualized)
    lo_le = lolp * 8760.0

    return {
        'lolp': lolp,
        'lolp_elec': lolp_elec,
        'lolp_heat': lolp_heat,
        'lolp_cold': lolp_cold,
        'lo_le_h_per_year': lo_le,
        'n_scenarios': n_scenarios,
        'total_hours': valid_hours,
    }


def compute_eens(lp_results, total_load_per_scenario=None, T=24):
    """EENS: Expected Energy Not Served.

    Definition:
      EENS (kWh/day)   = mean shed energy per scenario-day (absolute)
      EENS_norm (%)     = EENS / mean_total_load * 100% (cross-system comparable)

    EENS captures severity: a scenario with 500 kW of shed for 1 hour
    contributes 500 kWh, while 0.1 kW for 1 hour contributes 0.1 kWh.
    This continuous weighting is the key advantage over binary PFR.

    Args:
        lp_results: list of result dicts from IESOptimizer.solve()
        total_load_per_scenario: (N,) array of daily total load (elec+heat+cold, kWh)
                                 If None, EENS_norm is not computed.
        T: periods per day (24)

    Returns:
        dict with: eens_kw, eens_elec, eens_heat, eens_cold,
                   eens_norm_pct (if total_load provided)
    """
    n = len(lp_results)
    total_hours = n * T

    shed_by_hour = np.zeros(total_hours)
    shed_elec_by_hour = np.zeros(total_hours)
    shed_heat_by_hour = np.zeros(total_hours)
    shed_cold_by_hour = np.zeros(total_hours)

    idx = 0
    for r in lp_results:
        if not r.get('feasible', False):
            continue
        if 'per_hour' not in r:
            continue
        ph = r['per_hour']
        shed_by_hour[idx:idx+T]      = ph['shed']
        shed_elec_by_hour[idx:idx+T] = ph['shed_elec']
        shed_heat_by_hour[idx:idx+T] = ph['shed_heat']
        shed_cold_by_hour[idx:idx+T] = ph['shed_cold']
        idx += T

    valid_hours = idx
    if valid_hours == 0:
        return {
            'eens_kw': 0.0, 'eens_elec': 0.0,
            'eens_heat': 0.0, 'eens_cold': 0.0,
            'eens_norm_pct': 0.0, 'n_scenarios': n,
        }

    n_scenarios = valid_hours // T

    # EENS: total shed energy / number of scenarios (kWh per scenario-day)
    eens_total = float(shed_by_hour[:valid_hours].sum() / max(n_scenarios, 1))
    eens_elec  = float(shed_elec_by_hour[:valid_hours].sum() / max(n_scenarios, 1))
    eens_heat  = float(shed_heat_by_hour[:valid_hours].sum() / max(n_scenarios, 1))
    eens_cold  = float(shed_cold_by_hour[:valid_hours].sum() / max(n_scenarios, 1))

    result = {
        'eens_kw': eens_total,
        'eens_elec': eens_elec,
        'eens_heat': eens_heat,
        'eens_cold': eens_cold,
        'n_scenarios': n_scenarios,
    }

    # Normalized EENS (% of total load) — cross-system comparable
    if total_load_per_scenario is not None:
        mean_load = np.mean(total_load_per_scenario)
        result['eens_norm_pct'] = float(eens_total / max(mean_load, 1e-8) * 100.0)
    else:
        result['eens_norm_pct'] = 0.0

    return result


# ===========================================================================
# Slack Duration Curve
# ===========================================================================

def compute_slack_duration_data(lp_results, T=24):
    """Generate slack duration curve data for all three bus types.

    Slack Duration Curve — analogous to Load Duration Curve in power systems:
      X-axis: exceedance probability (0–100%)
      Y-axis: shed power (kW), sorted descending

    Shape interpretation:
      - Curve near zero across most of x-axis → high reliability
      - High values at small x (rare events) → heavy-tail shedding
      - Area under curve ∝ EENS

    Args:
        lp_results: list of result dicts from IESOptimizer.solve()
        T: periods per day (24)

    Returns:
        dict with: elec, heat, cold → sorted arrays of shed kW,
                   exceedance_pct → corresponding x-axis values (0–100%)
    """
    all_elec = []
    all_heat = []
    all_cold = []

    for r in lp_results:
        if not r.get('feasible', False) or 'per_hour' not in r:
            continue
        ph = r['per_hour']
        all_elec.extend(ph['shed_elec'].tolist())
        all_heat.extend(ph['shed_heat'].tolist())
        all_cold.extend(ph['shed_cold'].tolist())

    if not all_elec:
        return {bus: {'sorted_kw': np.array([]), 'exceedance_pct': np.array([])}
                for bus in ['elec', 'heat', 'cold']}

    data = {}
    for bus_name, values in [('elec', all_elec), ('heat', all_heat), ('cold', all_cold)]:
        arr = np.array(values)
        sorted_arr = np.sort(arr)[::-1]  # descending
        exceedance = np.arange(1, len(sorted_arr) + 1) / len(sorted_arr) * 100.0
        data[bus_name] = {
            'sorted_kw': sorted_arr,
            'exceedance_pct': exceedance,
        }

    return data


# ===========================================================================
# VSS — Proper 2-Stage Stochastic Dispatch
# ===========================================================================

def _compute_vss_kw(scenarios_kw, opt=None):
    """Value of Stochastic Solution — proper 2-stage stochastic programming.

    Framework (Birge & Louveaux, 2011):
      Stage 1 (day-ahead): GT output commitment — decided before uncertainty resolves
      Stage 2 (real-time): per-scenario re-dispatch — storage, grid, slack
                           optimized given fixed GT plan

      WS  = E_ξ[min_{x,y} f(x, y, ξ)]  (Wait-and-See: perfect foresight)
      EV  = min_{x,y} f(x, y, E[ξ])    (Expected Value: deterministic, mean scenario)
      EEV = E_ξ[f(x̄_EV, y*(ξ), ξ)]    (Expected result of EV: fix GT, re-dispatch)
      VSS = EEV - WS ≥ 0               (value of stochastic over deterministic)

    Key fix over physics_feasibility_metrics.compute_vss():
      EEV uses re-dispatch with GT FIXED to EV plan (not penalty-based approximation).
      This properly measures the cost of day-ahead GT commitment being misaligned
      with the actual scenario, without double-counting slack costs.

    Args:
        scenarios_kw: (N, 24, 5) array in kW [wind, solar, elec, heat, cold]
        opt: IESOptimizer instance (creates one if None)

    Returns:
        dict with: vss, vss_pct, ws_cost, eev_cost, ev_cost,
                   n_scenarios, n_eev_feasible
    """
    from ies_optimizer import IESOptimizer
    if opt is None:
        opt = IESOptimizer()

    N = scenarios_kw.shape[0]
    s = np.asarray(scenarios_kw)  # (N, 24, 5)

    # ===== WS (Wait-and-See): perfect foresight per scenario =====
    ws_results = opt.solve_batch(s)
    ws_costs = []
    for r in ws_results:
        if r['feasible'] and np.isfinite(r['total_cost']):
            ws_costs.append(r['total_cost'])

    if len(ws_costs) == 0:
        return {
            'vss': float('nan'), 'vss_pct': float('nan'),
            'ws_cost': float('inf'), 'eev_cost': float('inf'),
            'ev_cost': float('inf'), 'n_scenarios': N, 'n_eev_feasible': 0,
        }

    ws_cost = np.mean(ws_costs)

    # ===== EV (Expected Value): deterministic dispatch on mean scenario =====
    mean_scenario = s.mean(axis=0)  # (24, 5)
    ev_result = opt.solve(
        mean_scenario[:, 0], mean_scenario[:, 1],
        mean_scenario[:, 2], mean_scenario[:, 3], mean_scenario[:, 4],
        return_full=True,
    )

    if not ev_result['feasible'] or not np.isfinite(ev_result['total_cost']):
        return {
            'vss': float('nan'), 'vss_pct': float('nan'),
            'ws_cost': ws_cost, 'eev_cost': float('inf'),
            'ev_cost': float('inf'), 'n_scenarios': N, 'n_eev_feasible': 0,
        }

    ev_cost = ev_result['total_cost']
    gt_plan = ev_result['gt_plan']  # (24,) — GT output from EV solution

    # ===== EEV: re-dispatch each scenario with GT fixed to EV plan =====
    eev_costs = []
    for i in range(N):
        r = opt.solve_with_fixed_gt(
            s[i, :, 0], s[i, :, 1], s[i, :, 2],
            s[i, :, 3], s[i, :, 4], gt_plan,
        )
        if r['feasible'] and np.isfinite(r['total_cost']):
            eev_costs.append(r['total_cost'])

    if len(eev_costs) == 0:
        return {
            'vss': float('nan'), 'vss_pct': float('nan'),
            'ws_cost': ws_cost, 'eev_cost': float('inf'),
            'ev_cost': ev_cost, 'n_scenarios': N, 'n_eev_feasible': 0,
        }

    eev_cost = np.mean(eev_costs)

    vss = eev_cost - ws_cost
    vss_pct = vss / max(abs(ws_cost), 1e-8) * 100.0

    return {
        'vss': float(vss),
        'vss_pct': float(vss_pct),
        'ws_cost': float(ws_cost),
        'eev_cost': float(eev_cost),
        'ev_cost': float(ev_cost),
        'n_scenarios': N,
        'n_eev_feasible': len(eev_costs),
    }


# ===========================================================================
# 4-Layer Cost Decomposition
# ===========================================================================

def decompose_cost_4layer(result):
    """4-layer cost decomposition for IES dispatch result.

    Layer 1 (Day-ahead):     GT fuel + grid TOU purchase + O&M + carbon
                             → fixed costs from day-ahead unit commitment
    Layer 2 (Real-time):     Curtailment + heat dump
                             → adjustment costs from uncertainty realization
    Layer 3 (Demand Response): DR compensation (cheaper than shedding)
    Layer 4 (Infeasibility): Emergency load shedding penalty + excess energy
                             → direct measure of scenario quality

    Args:
        result: dict from IESOptimizer.solve() or solve_with_fixed_gt()

    Returns:
        dict with: day_ahead, real_time, dr, infeasibility, total,
                   breakdown_pct (each layer as % of total)
    """
    if not result.get('feasible', False):
        return {
            'day_ahead': float('inf'), 'real_time': float('inf'),
            'dr': float('inf'), 'infeasibility': float('inf'),
            'total': float('inf'), 'breakdown_pct': {},
        }

    cb = result['cost_breakdown']

    day_ahead    = cb['gas'] + cb['electricity'] + cb['om'] + cb['carbon']
    dr_cost      = cb.get('dr', 0.0)
    real_time    = cb.get('curtailment', 0.0) + cb.get('heat_dump', 0.0)
    infeasibility = cb['slack_penalty']
    total         = day_ahead + dr_cost + real_time + infeasibility

    return {
        'day_ahead': day_ahead,
        'real_time': real_time,
        'dr': dr_cost,
        'infeasibility': infeasibility,
        'total': total,
        'breakdown_pct': {
            'day_ahead': day_ahead / max(total, 1e-8) * 100.0,
            'real_time': real_time / max(total, 1e-8) * 100.0,
            'dr': dr_cost / max(total, 1e-8) * 100.0,
            'infeasibility': infeasibility / max(total, 1e-8) * 100.0,
        },
    }


# ===========================================================================
# Unified Reliability Evaluation
# ===========================================================================

class IESReliabilityEvaluator:
    """Unified IES reliability evaluation using IEEE 1366 LOLP/EENS + VSS.

    Usage:
        evaluator = IESReliabilityEvaluator(optimizer=opt)
        results = evaluator.evaluate(scenarios_kw)  # single batch
        table = evaluator.evaluate_models(model_scenarios_dict)  # multi-model
    """

    def __init__(self, optimizer=None):
        from ies_optimizer import IESOptimizer
        self.opt = optimizer or IESOptimizer()

    def evaluate(self, scenarios_kw):
        """Full reliability evaluation for a single batch of scenarios.

        Args:
            scenarios_kw: (N, 24, 5) array in kW [wind, solar, elec, heat, cold]

        Returns:
            dict with: lolp, eens, vss, cost_layers, slack_duration, pfr, n_scenarios
        """
        N = scenarios_kw.shape[0]

        # Run LP dispatch
        lp_results = self.opt.solve_batch(scenarios_kw)

        # PFR (legacy, for backward compatibility)
        feasible_arr = np.array([r['physically_feasible'] for r in lp_results])
        pfr = float(feasible_arr.mean()) * 100.0

        # IEEE 1366 reliability
        lolp = compute_lolp(lp_results)

        # Total load for normalized EENS
        s = np.asarray(scenarios_kw)
        total_load_per_scenario = s[:, :, 2:5].sum(axis=(1, 2))  # (N,) kWh/day
        eens = compute_eens(lp_results, total_load_per_scenario)

        # VSS — proper 2-stage
        vss = _compute_vss_kw(scenarios_kw, self.opt)

        # Cost decomposition (aggregate across scenarios)
        day_ahead_costs = []
        real_time_costs = []
        dr_costs = []
        infeas_costs = []
        total_costs_agg = []
        for r in lp_results:
            if r['feasible']:
                dec = decompose_cost_4layer(r)
                if np.isfinite(dec['day_ahead']):
                    day_ahead_costs.append(dec['day_ahead'])
                    real_time_costs.append(dec['real_time'])
                    dr_costs.append(dec['dr'])
                    infeas_costs.append(dec['infeasibility'])
                    total_costs_agg.append(dec['total'])

        cost_layers = {
            'day_ahead_mean': float(np.mean(day_ahead_costs)) if day_ahead_costs else float('inf'),
            'real_time_mean': float(np.mean(real_time_costs)) if real_time_costs else float('inf'),
            'dr_mean': float(np.mean(dr_costs)) if dr_costs else float('inf'),
            'infeasibility_mean': float(np.mean(infeas_costs)) if infeas_costs else float('inf'),
            'total_mean': float(np.mean(total_costs_agg)) if total_costs_agg else float('inf'),
        }

        # Slack duration curve data
        slack_dur = compute_slack_duration_data(lp_results)

        return {
            'pfr': pfr,
            'lolp': lolp,
            'eens': eens,
            'vss': vss,
            'cost_layers': cost_layers,
            'slack_duration': slack_dur,
            'n_scenarios': N,
            'lp_results': lp_results,  # raw results for further analysis
        }

    def evaluate_from_csv(self, csv_path, scaler):
        """Evaluate from a CSV file of generated scenarios.

        Args:
            csv_path: path to generated_scenarios_*.csv
            scaler: fitted MinMaxScaler for denormalization

        Returns:
            same dict as evaluate()
        """
        import pandas as pd
        from dataset_energy_v3 import FEATURE_COLS

        df = pd.read_csv(csv_path, encoding='utf-8-sig')
        raw_vals = df[FEATURE_COLS].values.astype(np.float32)
        norm_vals = scaler.transform(raw_vals)

        N = len(norm_vals) // 24
        if N == 0:
            return {'error': 'No scenarios found', 'n_scenarios': 0}

        scenarios_norm = norm_vals.reshape(N, 24, 5)

        # Denormalize to kW
        flat = scenarios_norm.reshape(-1, 5)
        denormed = scaler.inverse_transform(flat)
        scenarios_kw = denormed.reshape(N, 24, 5)

        return self.evaluate(scenarios_kw)


# ===========================================================================
# Backward-compatible drop-in replacements for physics_feasibility_metrics.py
# These match the old API signatures so existing callers work unchanged.
# ===========================================================================

_evaluator = None

def _get_evaluator():
    global _evaluator
    if _evaluator is None:
        _evaluator = IESReliabilityEvaluator()
    return _evaluator


def evaluate_physics_feasibility(normalized_scenarios, scaler=None):
    """Drop-in replacement for physics_feasibility_metrics.evaluate_physics_feasibility().

    Returns the OLD dict structure for backward compatibility, PLUS new
    reliability fields (lolp, eens, vss_new, cost_layers).

    Args:
        normalized_scenarios: (N, T, C) in [0, 1]
        scaler: MinMaxScaler for denormalization

    Returns:
        dict with original keys + reliability keys
    """
    evaluator = _get_evaluator()

    N = normalized_scenarios.shape[0]
    if scaler is not None:
        flat = normalized_scenarios.reshape(-1, 5)
        denormed = scaler.inverse_transform(flat)
        scenarios_kw = denormed.reshape(N, 24, 5)
    else:
        scenarios_kw = normalized_scenarios.copy()

    rel = evaluator.evaluate(scenarios_kw)

    # Build old-style daily_operating_cost dict
    daily_operating_cost = {
        'per_scenario': np.full(N, rel['cost_layers']['total_mean']),
        'per_scenario_slack': np.full(N, rel['cost_layers']['infeasibility_mean']),
        'per_scenario_breakdown': [],
        'mean': rel['cost_layers']['total_mean'],
        'std': 0.0,
        'median': rel['cost_layers']['total_mean'],
        'mean_slack': rel['cost_layers']['infeasibility_mean'],
        'slack_ratio': rel['cost_layers']['infeasibility_mean'] / max(rel['cost_layers']['total_mean'], 1e-8) * 100.0,
        'lp_feasible_rate': rel['pfr'],
        'lp_feasible_count': int(rel['pfr'] / 100.0 * N),
        'lp_total_scenarios': N,
        'mean_cost_breakdown': {},
    }

    return {
        # Original keys (backward compatible)
        'feasible_rate': rel['pfr'],
        'feasible_count': int(rel['pfr'] / 100.0 * N),
        'total_scenarios': N,
        'daily_operating_cost': daily_operating_cost,
        # New reliability keys
        'lolp': rel['lolp'],
        'eens': rel['eens'],
        'vss_reliability': rel['vss'],
        'cost_layers': rel['cost_layers'],
    }


def compute_vss(normalized_scenarios, scaler=None):
    """Drop-in replacement for physics_feasibility_metrics.compute_vss().

    Uses the PROPER 2-stage re-dispatch approach (not penalty-based).

    Args:
        normalized_scenarios: (N, T, C) in [0, 1] or kW if scaler is None
        scaler: MinMaxScaler for denormalization (or None if already in kW)

    Returns:
        dict with: vss, vss_relative, ws_cost, eev_cost, ev_cost, n_scenarios
    """
    evaluator = _get_evaluator()

    N = normalized_scenarios.shape[0]
    if scaler is not None:
        flat = normalized_scenarios.reshape(-1, 5)
        denormed = scaler.inverse_transform(flat)
        scenarios_kw = denormed.reshape(N, 24, 5)
    else:
        scenarios_kw = normalized_scenarios.copy()

    vss_result = _compute_vss_kw(scenarios_kw, evaluator.opt)

    return {
        'vss': vss_result['vss'],
        'vss_relative': vss_result['vss_pct'],
        'ws_cost': vss_result['ws_cost'],
        'eev_cost': vss_result['eev_cost'],
        'ev_cost': vss_result['ev_cost'],
        'n_scenarios': vss_result['n_scenarios'],
    }


# ===========================================================================
# Publication-Quality Plotting
# ===========================================================================

# rcParams for reliability figures
_RELIABILITY_RCPARAMS = {
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'DejaVu Sans'],
    'axes.unicode_minus': False,
    'font.size': 9, 'axes.titlesize': 11, 'axes.labelsize': 10,
    'legend.fontsize': 7.5, 'xtick.labelsize': 8, 'ytick.labelsize': 8,
    'axes.linewidth': 0.8,
    'xtick.major.size': 4, 'xtick.major.width': 0.7,
    'ytick.major.size': 4, 'ytick.major.width': 0.7,
    'figure.dpi': 150, 'savefig.dpi': 300,
    'axes.spines.top': False, 'axes.spines.right': False,
    'grid.alpha': 0.12, 'grid.linewidth': 0.4,
}

# Color scheme consistent with paper's other figures
_MODEL_COLORS = {
    'ConvCVAE': '#D55E00',
    'WGAN-GP':  '#CC79A7',
    'ED-DDPM':  '#56B4E9',
    'DC-DDPM':  '#009E73',
    'DC-CAF':   '#4477AA',
    'STAR-Flow':'#E69F00',
    'IESGAT':   '#AA4499',
    'Real Data':'#333333',
}
_MODEL_LS = {
    'ConvCVAE': '--',
    'WGAN-GP':  ':',
    'ED-DDPM':  '-',
    'DC-DDPM':  '-',
    'DC-CAF':   '-',
    'STAR-Flow':'--',
    'IESGAT':   '-.',
    'Real Data': (0, (3, 1, 1, 1)),
}
_MODEL_LW = {
    'ConvCVAE': 1.2,
    'WGAN-GP':  1.2,
    'ED-DDPM':  1.5,
    'DC-DDPM':  1.5,
    'DC-CAF':   1.5,
    'STAR-Flow': 2.5,
    'IESGAT':   1.5,
    'Real Data': 2.0,
}


def plot_reliability_figure(all_model_results, save_dir='images',
                             filename_prefix='fig_reliability'):
    """Generate publication-quality 3-panel reliability figure.

    Panel (a): Slack Duration Curves (all models + real data, per bus)
    Panel (b): LOLP vs EENS scatter (one point per model, ideal = origin)
    Panel (c): 4-Layer Cost Decomposition (stacked bar per model)

    Args:
        all_model_results: dict of {model_label: evaluate() output}
        save_dir: output directory
        filename_prefix: base filename for saved figures

    Saves:
        {save_dir}/{filename_prefix}.pdf and .png
    """
    plt.rcParams.update(_RELIABILITY_RCPARAMS)

    fig = plt.figure(figsize=(16, 5.5))

    # ---- Panel (a): Slack Duration Curves (3 buses) ----
    bus_names = ['Electric', 'Heat', 'Cold']
    bus_keys  = ['elec', 'heat', 'cold']

    for col, (bus_name, bus_key) in enumerate(zip(bus_names, bus_keys)):
        ax = plt.subplot(1, 3, col + 1)

        for model_label, eval_result in all_model_results.items():
            sd = eval_result.get('slack_duration', {})
            bus_data = sd.get(bus_key, {})
            sorted_kw = bus_data.get('sorted_kw', np.array([]))
            exceedance = bus_data.get('exceedance_pct', np.array([]))

            if len(sorted_kw) == 0:
                continue

            color = _MODEL_COLORS.get(model_label, '#888888')
            ls    = _MODEL_LS.get(model_label, '-')
            lw    = _MODEL_LW.get(model_label, 1.5)

            # Filter to non-zero values for cleaner visualization
            nonzero_mask = sorted_kw > 0.1
            if nonzero_mask.sum() > 0:
                ax.plot(exceedance[nonzero_mask], sorted_kw[nonzero_mask],
                        color=color, ls=ls, lw=lw,
                        label=model_label if col == 2 else '')
            else:
                # Plot a zero-line if no shedding
                ax.plot([0, 100], [0, 0], color=color, ls=ls, lw=lw,
                        label=model_label if col == 2 else '')

        ax.set_xlabel('Exceedance Probability (%)')
        ax.set_ylabel(f'{bus_name} Shedding (kW)' if col == 0 else '')
        ax.set_title(f'({chr(97 + col)}) {bus_name} Bus', fontweight='bold')
        ax.set_xlim(0, 100)
        ax.set_yscale('symlog', linthresh=1.0)  # log scale above 1 kW
        ax.grid(True, alpha=0.12)
        ax.tick_params(labelsize=7.5)

    # Shared legend above panel (a)
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc='upper center', ncol=len(handles),
                   fontsize=8, frameon=True, framealpha=0.9,
                   bbox_to_anchor=(0.5, 0.99))

    plt.tight_layout(rect=[0, 0, 1, 0.92])

    for fmt in ['pdf', 'png']:
        path = os.path.join(save_dir, f'{filename_prefix}_slack_duration.{fmt}')
        fig.savefig(path, format=fmt, dpi=300, bbox_inches='tight', pad_inches=0.15)
        print(f'Saved: {path}')
    plt.close(fig)

    # ---- Panel (b): LOLP vs EENS scatter ----
    fig2, ax2 = plt.subplots(figsize=(7, 5.5))
    plt.rcParams.update(_RELIABILITY_RCPARAMS)

    for model_label, eval_result in all_model_results.items():
        lolp_val = eval_result.get('lolp', {}).get('lolp', 0) * 100  # convert to %
        eens_val = eval_result.get('eens', {}).get('eens_kw', 0)

        color = _MODEL_COLORS.get(model_label, '#888888')
        marker = 's' if model_label == 'Real Data' else 'o'

        ax2.scatter(eens_val, lolp_val, c=color, marker=marker,
                    s=120, edgecolors='white', linewidth=0.8, zorder=5,
                    label=model_label)
        ax2.annotate(model_label, (eens_val, lolp_val),
                     textcoords="offset points", xytext=(5, 5),
                     fontsize=7, color=color, alpha=0.9)

    ax2.set_xlabel('EENS (kWh/day)')
    ax2.set_ylabel('LOLP (%)')
    ax2.set_title('(d) Reliability Trade-off: LOLP vs EENS', fontweight='bold')
    ax2.grid(True, alpha=0.12)
    ax2.legend(fontsize=7, loc='upper left', framealpha=0.9)

    plt.tight_layout()
    for fmt in ['pdf', 'png']:
        path = os.path.join(save_dir, f'{filename_prefix}_lolp_eens.{fmt}')
        fig2.savefig(path, format=fmt, dpi=300, bbox_inches='tight', pad_inches=0.15)
        print(f'Saved: {path}')
    plt.close(fig2)

    print('Reliability figures complete!')


# ===========================================================================
# Self-Test
# ===========================================================================

if __name__ == '__main__':
    import pandas as pd
    from sklearn.preprocessing import MinMaxScaler
    from dataset_energy_v3 import load_energy_data

    print("=" * 65)
    print("IES Reliability Metrics Self-Test (IEEE 1366 LOLP/EENS)")
    print("=" * 65)

    DATA_PATH = "./源荷数据集.csv"

    if os.path.exists(DATA_PATH):
        # Load real data
        raw_data, _ = load_energy_data(DATA_PATH)
        full = raw_data.astype(np.float32)
        scaler = MinMaxScaler()
        scaler.fit(full)

        # Real data baseline
        from ies_optimizer import IESOptimizer
        opt = IESOptimizer()
        evaluator = IESReliabilityEvaluator(optimizer=opt)

        real_days_kw = full.reshape(365, 24, 5)

        print("\n[1] Real Data (365 days) Reliability")
        real_eval = evaluator.evaluate(real_days_kw)
        print(f"  PFR:           {real_eval['pfr']:.1f}%")
        print(f"  LOLP:          {real_eval['lolp']['lolp']*100:.2f}% "
              f"(LOLE = {real_eval['lolp']['lo_le_h_per_year']:.1f} h/yr)")
        print(f"  LOLP-elec:     {real_eval['lolp']['lolp_elec']*100:.2f}%")
        print(f"  LOLP-heat:     {real_eval['lolp']['lolp_heat']*100:.2f}%")
        print(f"  LOLP-cold:     {real_eval['lolp']['lolp_cold']*100:.2f}%")
        print(f"  EENS:          {real_eval['eens']['eens_kw']:.1f} kWh/day")
        print(f"  EENS (norm):   {real_eval['eens']['eens_norm_pct']:.2f}%")
        print(f"  EENS-elec:     {real_eval['eens']['eens_elec']:.1f} kWh/day")
        print(f"  EENS-heat:     {real_eval['eens']['eens_heat']:.1f} kWh/day")
        print(f"  EENS-cold:     {real_eval['eens']['eens_cold']:.1f} kWh/day")

        # Cost layers
        cl = real_eval['cost_layers']
        print(f"\n  4-Layer Cost Decomposition (daily avg):")
        print(f"    Day-ahead:        {cl['day_ahead_mean']:.1f} yuan")
        print(f"    Real-time:        {cl['real_time_mean']:.1f} yuan")
        print(f"    Demand Response:  {cl['dr_mean']:.1f} yuan")
        print(f"    Infeasibility:    {cl['infeasibility_mean']:.1f} yuan")
        print(f"    Total:            {cl['total_mean']:.1f} yuan")

        # VSS on real data (sample)
        print(f"\n[2] VSS (real data, first 50 days)")
        vss = _compute_vss_kw(real_days_kw[:50], opt)
        print(f"  WS:    {vss['ws_cost']:.1f} yuan/day")
        print(f"  EV:    {vss['ev_cost']:.1f} yuan/day")
        print(f"  EEV:   {vss['eev_cost']:.1f} yuan/day")
        print(f"  VSS:   {vss['vss']:.1f} yuan/day ({vss['vss_pct']:.1f}%)")
        print(f"  EEV feasible: {vss['n_eev_feasible']}/{vss['n_scenarios']}")

        # Test with a generated model CSV if available
        csv_test = 'results_energy_continuous_v3_fm/generated_scenarios_v3_fm_all12.csv'
        if os.path.exists(csv_test):
            print(f"\n[3] STAR-Flow (V3-FM) from {csv_test}")
            starflow_eval = evaluator.evaluate_from_csv(csv_test, scaler)
            print(f"  PFR:           {starflow_eval['pfr']:.1f}%")
            print(f"  LOLP:          {starflow_eval['lolp']['lolp']*100:.2f}%")
            print(f"  EENS:          {starflow_eval['eens']['eens_kw']:.1f} kWh/day")
            print(f"  EENS (norm):   {starflow_eval['eens']['eens_norm_pct']:.2f}%")
            print(f"  VSS:           {starflow_eval['vss']['vss']:.1f} yuan/day "
                  f"({starflow_eval['vss']['vss_pct']:.1f}%)")

        # VSS — proper 2-stage (no comparison needed, old penalty-based was buggy)
        print(f"\n[4] VSS (proper 2-stage, 50 real days)")
        vss_new = _compute_vss_kw(real_days_kw[:50], opt)
        print(f"  WS:    {vss_new['ws_cost']:.1f} yuan/day")
        print(f"  EV:    {vss_new['ev_cost']:.1f} yuan/day")
        print(f"  EEV:   {vss_new['eev_cost']:.1f} yuan/day")
        print(f"  VSS:   {vss_new['vss']:.1f} yuan/day ({vss_new['vss_pct']:.1f}%)")

    print(f"\n{'=' * 65}")
    print("Self-test complete!")
