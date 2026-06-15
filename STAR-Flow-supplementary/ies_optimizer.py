"""
IES LP Optimizer with Slack Variables for Physical Feasibility Verification.

Core IES model (simplified from refinedP2G.m, DOC Chapter 3):
  9 devices: GT, WHB, GB, EB, EC, AC, BT(battery), HS(thermal storage), Grid

Key innovation: Slack variables on all 3 power balance equations.
  - Physically feasible scenario -> slack=0 -> normal cost
  - Physically infeasible scenario -> slack>0 -> normal cost + M * slack_penalty
  - Slack penalty directly monetizes the cost of physical infeasibility

Uses scipy.optimize.linprog (HiGHS) — guaranteed available, no extra install needed.
"""
import numpy as np
from scipy.optimize import linprog

# ===========================================================================
# IES Device Parameters (from refinedP2G.m)
# ===========================================================================
T = 24  # periods per day

# -- Gas Turbine (GT) --  (DOC Table 3-2: P_GT,max=500)
ETA_GT   = 0.35
P_GT_MIN = 125   # 25% of rated capacity (standard: 25-40%)
P_GT_MAX = 500
P_GT_RAMP = 250

# -- Waste Heat Boiler (WHB), coupled to GT --
# Q_WHB = P_GT * (1 - ETA_GT - 0.15) * ETA_REC / ETA_GT
ETA_REC  = 0.8
WHB_COEF = (1 - ETA_GT - 0.15) * ETA_REC / ETA_GT  # = 1.142857...

# -- Gas Boiler (GB) -- supplementary gas-to-heat (DOC-aligned, matches EB capacity)
ETA_GB   = 0.8
Q_GB_MIN = 0
Q_GB_MAX = 250
Q_GB_RAMP = 250

# -- Electric Boiler (EB) --  (DOC Table 3-2: H_EB,max=250)
ETA_EB   = 0.85
Q_EB_MIN = 0
Q_EB_MAX = 250
Q_EB_RAMP = 250

# -- Electric Chiller (EC) --  (DOC: 300, expanded to 650 for P90 cold coverage)
COP_EC   = 4.0
C_EC_MIN = 0
C_EC_MAX = 650
C_EC_RAMP = 650

# -- Absorption Chiller (AC) --  (DOC: 300, expanded to 500 for P90 cold coverage)
COP_AC   = 1.2
C_AC_MIN = 0
C_AC_MAX = 500
C_AC_RAMP = 500

# -- Battery Storage (BT) --
ETA_BT_CH  = 0.95
ETA_BT_DIS = 0.95
E_BT_MAX   = 400
E_BT0      = 200
P_BT_CH_MAX = 100
P_BT_DIS_MAX = 100
P_BT_RAMP   = 100
E_BT_MIN = 0.1 * E_BT_MAX  # 40
E_BT_MAX_SOC = 0.9 * E_BT_MAX  # 360

# -- Thermal Storage (HS) --
ETA_HS_CH  = 0.9
ETA_HS_DIS = 0.9
ETA_HS_LOSS = 0.01
H_HS_MAX   = 400
H_HS0      = 200
Q_HS_CH_MAX = 100
Q_HS_DIS_MAX = 100
Q_HS_RAMP   = 100
H_HS_MIN   = 0.1 * H_HS_MAX  # 40
H_HS_MAX_SOC = 0.9 * H_HS_MAX  # 360

# -- Grid --
P_GRID_MAX = 1500

# -- Natural gas --
L_CH4 = 9.7     # kWh/m3
C_GAS = 3.5     # yuan/m3

# -- Electricity price (TOU, 24h) --
C_BUY = np.array([0.48, 0.48, 0.48, 0.48, 0.48, 0.48, 0.48,
                   0.88, 0.88, 0.88, 0.88, 1.10, 1.10, 1.10,
                   0.88, 0.88, 0.88, 0.88, 1.10, 1.10, 1.10,
                   1.10, 0.48, 0.48])

# -- O&M costs (yuan/kWh) --
C_OM_GT  = 0.0472
C_OM_WHB = 0.00216
C_OM_GB  = 0.0216
C_OM_EB  = 0.0087
C_OM_EC  = 0.0072
C_OM_AC  = 0.0087
C_OM_BT  = 0.018
C_OM_HS  = 0.016
C_OM_WT  = 0.0196
C_OM_PV  = 0.0196

# -- Carbon --
DELTA_CO2 = 1.02 - 0.728  # grid CO2 emission factor (kg/kWh)
C_CO2     = 0.14           # carbon price (yuan/kg)

# -- Curtailment penalty --
C_CURTAIL = 1.0   # yuan/kWh

# -- Slack penalties (3-tier, aligned with CDSG and IEEE standards) --
# Load shedding (S_*_p): failing to meet demand → emergency, expensive
#   ~10x peak TOU price, represents emergency load shedding or reserve activation
C_SHED_ELEC = 10.0   # yuan/kWh — electric load shedding
C_SHED_HEAT = 8.0    # yuan/kWh — heat load shedding
C_SHED_COLD = 8.0    # yuan/kWh — cold load shedding

# Excess generation (S_*_n): surplus beyond storage/conversion capacity → cheap
#   Minor administrative cost, already partially covered by curtailment penalty
C_EXCESS = 0.5       # yuan/kWh — all energy types

# Waste heat dump (environmental cost for dumping GT exhaust heat)
C_HEAT_DUMP = 0.1     # yuan/kWh — minor environmental penalty

# -- Demand Response (DR) --
# Load curtailment: cheaper than emergency shedding, more expensive than normal dispatch
# Cost hierarchy: normal dispatch (0.5-1.0) < DR (2.5-3.0) < load shedding (8-10) yuan/kWh
DR_MAX_ELEC = 0.10    # max 10% of electric load curtailable
DR_MAX_HEAT = 0.10    # max 10% of heat load curtailable
DR_MAX_COLD = 0.10    # max 10% of cold load curtailable
C_DR_ELEC = 3.0       # yuan/kWh — DR compensation (electric)
C_DR_HEAT = 2.5       # yuan/kWh — DR compensation (heat)
C_DR_COLD = 2.5       # yuan/kWh — DR compensation (cold)

# ===========================================================================
# Variable Layout (each block has T=24 entries, total N_VARS = 28 * 24 = 672)
# ===========================================================================
VAR_NAMES = [
    'P_gt', 'P_grid', 'P_wind', 'P_pv',            # 0-3
    'P_eb', 'Q_eb', 'P_ec', 'C_ec',                 # 4-7
    'Q_gb', 'Q_whb', 'Q_ac', 'C_ac',                # 8-11
    'P_bt_ch', 'P_bt_dis', 'Q_hs_ch', 'Q_hs_dis',   # 12-15
    'E_bt', 'H_hs',                                  # 16-17
    'S_elec_p', 'S_elec_n', 'S_heat_p', 'S_heat_n', # 18-21
    'S_cold_p', 'S_cold_n',                          # 22-23
    'Q_dump',                                        # 24 — WHB waste heat dump
    'DR_elec', 'DR_heat', 'DR_cold',                 # 25-27 — Demand Response
]
N_VAR_BLOCKS = len(VAR_NAMES)
N_VARS = N_VAR_BLOCKS * T  # 672

def vidx(name, t=None):
    """Return variable index/indices for given name and optional time t."""
    base = VAR_NAMES.index(name) * T
    if t is None:
        return slice(base, base + T)
    return base + t

def _scalar(val):
    """Ensure scalar output from linprog result access."""
    if hasattr(val, 'item'):
        return val.item()
    return float(val)

# ===========================================================================
# LP Builder
# ===========================================================================

class IESOptimizer:
    """IES dispatch LP solver using scipy's HiGHS."""

    def __init__(self):
        pass

    # ------------------------------------------------------------------
    def _build_lp(self, P_wind_avail, P_pv_avail, P_load, H_load, C_load,
                   fixed_gt=None):
        """Build LP matrices and solve.

        Args:
            fixed_gt: optional length-24 array. When provided, P_gt is fixed
                      to these values (tight bounds, no ramp) for 2-stage VSS.

        Returns (success, result_dict)
        """
        # ---- Bounds (all variables >= 0 by default) ----
        bounds = [(0, None)] * N_VARS

        # Device capacity bounds
        for t in range(T):
            if fixed_gt is not None:
                bounds[vidx('P_gt', t)] = (fixed_gt[t], fixed_gt[t])  # fixed
            else:
                bounds[vidx('P_gt', t)] = (P_GT_MIN, P_GT_MAX)
            bounds[vidx('P_grid', t)]  = (0, P_GRID_MAX)
            bounds[vidx('P_wind', t)]  = (0, P_wind_avail[t])
            bounds[vidx('P_pv', t)]    = (0, P_pv_avail[t])
            bounds[vidx('Q_eb', t)]    = (Q_EB_MIN, Q_EB_MAX)
            bounds[vidx('C_ec', t)]    = (C_EC_MIN, C_EC_MAX)
            bounds[vidx('Q_gb', t)]    = (Q_GB_MIN, Q_GB_MAX)
            bounds[vidx('C_ac', t)]    = (C_AC_MIN, C_AC_MAX)
            bounds[vidx('P_bt_ch', t)] = (0, P_BT_CH_MAX)
            bounds[vidx('P_bt_dis', t)] = (0, P_BT_DIS_MAX)
            bounds[vidx('Q_hs_ch', t)]  = (0, Q_HS_CH_MAX)
            bounds[vidx('Q_hs_dis', t)] = (0, Q_HS_DIS_MAX)
            bounds[vidx('E_bt', t)]    = (E_BT_MIN, E_BT_MAX_SOC)
            bounds[vidx('H_hs', t)]    = (H_HS_MIN, H_HS_MAX_SOC)
            bounds[vidx('Q_dump', t)]  = (0, None)   # heat dump >= 0
            # DR: disabled for clean model comparison
            bounds[vidx('DR_elec', t)] = (0, 0)
            bounds[vidx('DR_heat', t)] = (0, 0)
            bounds[vidx('DR_cold', t)] = (0, 0)

        # Relaxed final SOC: 30%-70% range (Li et al. 2024, IEEE TSG)
        E_BT_MIN_FINAL = 0.3 * E_BT_MAX
        E_BT_MAX_FINAL = 0.7 * E_BT_MAX
        H_HS_MIN_FINAL = 0.3 * H_HS_MAX
        H_HS_MAX_FINAL = 0.7 * H_HS_MAX
        bounds[vidx('E_bt', T-1)] = (E_BT_MIN_FINAL, E_BT_MAX_FINAL)
        bounds[vidx('H_hs', T-1)] = (H_HS_MIN_FINAL, H_HS_MAX_FINAL)

        # ---- Equality constraints (A_eq @ x = b_eq) ----
        A_eq_rows = []
        b_eq_vals = []

        def add_row(row_dict, rhs):
            """Add constraint row: sum of named vars = rhs.
            row_dict: {var_name: {t: coeff, ...}} or {var_name: scalar_coeff_for_all_t}
            """
            row = np.zeros(N_VARS)
            for name, coeff in row_dict.items():
                if isinstance(coeff, dict):
                    for t, val in coeff.items():
                        row[vidx(name, t)] = val
                else:
                    row[vidx(name)] = coeff
            A_eq_rows.append(row)
            b_eq_vals.append(rhs)

        # Device coupling equalities (per time-step)
        for t in range(T):
            # WHB with heat dump: Q_whb + Q_dump = P_gt * WHB_COEF
            # GT exhaust heat can be recovered (Q_whb) or dumped (Q_dump)
            add_row({'Q_whb': {t: 1.0}, 'Q_dump': {t: 1.0},
                     'P_gt': {t: -WHB_COEF}}, 0)
            add_row({'Q_eb': {t: 1.0}, 'P_eb': {t: -ETA_EB}}, 0)            # EB
            add_row({'C_ec': {t: 1.0}, 'P_ec': {t: -COP_EC}}, 0)            # EC
            add_row({'C_ac': {t: 1.0}, 'Q_ac': {t: -COP_AC}}, 0)            # AC

        # Battery SOC dynamics
        add_row({'E_bt': {0: 1.0}, 'P_bt_ch': {0: -ETA_BT_CH},
                 'P_bt_dis': {0: 1.0/ETA_BT_DIS}}, E_BT0)
        for t in range(1, T):
            add_row({'E_bt': {t: 1.0, t-1: -1.0},
                     'P_bt_ch': {t: -ETA_BT_CH},
                     'P_bt_dis': {t: 1.0/ETA_BT_DIS}}, 0)
        # Battery final SOC: relaxed to 30%-70% range (Li et al. 2024, IEEE TSG)
        # Removed rigid equality E_bt[T-1] = E_BT0

        # Thermal storage SOC dynamics
        add_row({'H_hs': {0: 1.0}, 'Q_hs_ch': {0: -ETA_HS_CH},
                 'Q_hs_dis': {0: 1.0/ETA_HS_DIS}}, H_HS0)
        for t in range(1, T):
            add_row({'H_hs': {t: 1.0, t-1: -(1-ETA_HS_LOSS)},
                     'Q_hs_ch': {t: -ETA_HS_CH},
                     'Q_hs_dis': {t: 1.0/ETA_HS_DIS}}, 0)
        # Thermal final SOC: relaxed to 30%-70% range

        # Three power balances (with slack) — per time-step
        for t in range(T):
            # Electric: gen + slack_p - slack_n + DR = load + consumption
            add_row({
                'P_gt': {t: 1.0}, 'P_grid': {t: 1.0}, 'P_bt_dis': {t: 1.0},
                'P_wind': {t: 1.0}, 'P_pv': {t: 1.0},
                'S_elec_p': {t: 1.0}, 'S_elec_n': {t: -1.0},
                'DR_elec': {t: 1.0},
                'P_eb': {t: -1.0}, 'P_ec': {t: -1.0}, 'P_bt_ch': {t: -1.0},
            }, P_load[t])

            # Heat: gen + slack_p - slack_n + DR = load + consumption
            add_row({
                'Q_whb': {t: 1.0}, 'Q_gb': {t: 1.0}, 'Q_eb': {t: 1.0},
                'Q_hs_dis': {t: 1.0},
                'S_heat_p': {t: 1.0}, 'S_heat_n': {t: -1.0},
                'DR_heat': {t: 1.0},
                'Q_ac': {t: -1.0}, 'Q_hs_ch': {t: -1.0},
            }, H_load[t])

            # Cold: gen + slack_p - slack_n + DR = load
            add_row({
                'C_ec': {t: 1.0}, 'C_ac': {t: 1.0},
                'S_cold_p': {t: 1.0}, 'S_cold_n': {t: -1.0},
                'DR_cold': {t: 1.0},
            }, C_load[t])

        A_eq = np.array(A_eq_rows)
        b_eq = np.array(b_eq_vals)

        # ---- Inequality constraints (A_ub @ x <= b_ub) ----
        A_ub_rows = []
        b_ub_vals = []

        def add_ramp(var_name, ramp_limit):
            """Add ramp constraints: |x[t+1] - x[t]| <= ramp_limit for t=0..T-2."""
            for t in range(T - 1):
                row = np.zeros(N_VARS)
                row[vidx(var_name, t+1)] = 1.0
                row[vidx(var_name, t)] = -1.0
                A_ub_rows.append(row)
                b_ub_vals.append(ramp_limit)
                # Reverse direction
                row2 = np.zeros(N_VARS)
                row2[vidx(var_name, t)] = 1.0
                row2[vidx(var_name, t+1)] = -1.0
                A_ub_rows.append(row2)
                b_ub_vals.append(ramp_limit)

        def add_net_ramp(pos_var, neg_var, ramp_limit):
            """Ramp on net: |(pos[t+1]-neg[t+1]) - (pos[t]-neg[t])| <= ramp_limit."""
            for t in range(T - 1):
                row = np.zeros(N_VARS)
                row[vidx(pos_var, t+1)] = 1.0
                row[vidx(neg_var, t+1)] = -1.0
                row[vidx(pos_var, t)] = -1.0
                row[vidx(neg_var, t)] = 1.0
                A_ub_rows.append(row)
                b_ub_vals.append(ramp_limit)
                # Reverse
                row2 = np.zeros(N_VARS)
                row2[vidx(pos_var, t)] = 1.0
                row2[vidx(neg_var, t)] = -1.0
                row2[vidx(pos_var, t+1)] = -1.0
                row2[vidx(neg_var, t+1)] = 1.0
                A_ub_rows.append(row2)
                b_ub_vals.append(ramp_limit)

        if fixed_gt is None:
            add_ramp('P_gt', P_GT_RAMP)  # skip when GT is fixed (VSS Stage 2)
        add_ramp('Q_gb', Q_GB_RAMP)
        add_ramp('Q_eb', Q_EB_RAMP)
        add_ramp('C_ec', C_EC_RAMP)
        add_ramp('C_ac', C_AC_RAMP)
        add_net_ramp('P_bt_ch', 'P_bt_dis', P_BT_RAMP)
        add_net_ramp('Q_hs_ch', 'Q_hs_dis', Q_HS_RAMP)

        A_ub = np.array(A_ub_rows) if A_ub_rows else np.empty((0, N_VARS))
        b_ub = np.array(b_ub_vals)

        # ---- Objective function ----
        c = np.zeros(N_VARS)

        # Gas cost: C_GAS * (P_gt/ETA_GT + Q_gb/ETA_GB) / L_CH4 per period
        gas_coef_gt = C_GAS / (ETA_GT * L_CH4)
        gas_coef_gb = C_GAS / (ETA_GB * L_CH4)
        for t in range(T):
            c[vidx('P_gt', t)] += gas_coef_gt
            c[vidx('Q_gb', t)] += gas_coef_gb

        # Electricity cost: C_BUY[t] * P_grid[t]
        for t in range(T):
            c[vidx('P_grid', t)] += C_BUY[t]

        # O&M costs
        for t in range(T):
            c[vidx('P_gt', t)] += C_OM_GT
            c[vidx('Q_whb', t)] += C_OM_WHB
            c[vidx('Q_gb', t)] += C_OM_GB
            c[vidx('Q_eb', t)] += C_OM_EB
            c[vidx('C_ec', t)] += C_OM_EC
            c[vidx('C_ac', t)] += C_OM_AC
            c[vidx('P_bt_ch', t)] += C_OM_BT
            c[vidx('P_bt_dis', t)] += C_OM_BT
            c[vidx('Q_hs_ch', t)] += C_OM_HS
            c[vidx('Q_hs_dis', t)] += C_OM_HS
            c[vidx('P_wind', t)] += C_OM_WT
            c[vidx('P_pv', t)] += C_OM_PV

        # Carbon cost: DELTA_CO2 * C_CO2 * P_grid[t]
        for t in range(T):
            c[vidx('P_grid', t)] += DELTA_CO2 * C_CO2

        # Curtailment penalty: C_CURTAIL * (P_wind_avail - P_wind + P_pv_avail - P_pv)
        # = C_CURTAIL * (-P_wind - P_pv) + constant
        for t in range(T):
            c[vidx('P_wind', t)] += -C_CURTAIL
            c[vidx('P_pv', t)] += -C_CURTAIL

        # Slack penalties: differentiated by type
        # Load shedding (S_*_p): expensive — represents emergency measures
        for t in range(T):
            c[vidx('S_elec_p', t)] += C_SHED_ELEC
            c[vidx('S_heat_p', t)] += C_SHED_HEAT
            c[vidx('S_cold_p', t)] += C_SHED_COLD

        # Excess generation (S_*_n): cheap — minor curtailment
        for t in range(T):
            c[vidx('S_elec_n', t)] += C_EXCESS
            c[vidx('S_heat_n', t)] += C_EXCESS
            c[vidx('S_cold_n', t)] += C_EXCESS

        # Waste heat dump: minor environmental penalty
        for t in range(T):
            c[vidx('Q_dump', t)] += C_HEAT_DUMP

        # Demand Response: between normal dispatch and load shedding
        # Cost hierarchy: dispatch(~1) < DR(~2.5-3) < shedding(~8-10) yuan/kWh
        for t in range(T):
            c[vidx('DR_elec', t)] += C_DR_ELEC
            c[vidx('DR_heat', t)] += C_DR_HEAT
            c[vidx('DR_cold', t)] += C_DR_COLD

        # ---- Solve ----
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                      bounds=bounds, method='highs')

        return res

    # ------------------------------------------------------------------
    def solve(self, wind, solar, elec_load, heat_load, cold_load,
              return_full=False):
        """Solve IES dispatch for a single 24h scenario.

        Args:
            wind, solar, elec_load, heat_load, cold_load: length-24 arrays (kW)
            return_full: if True, include x_solution and gt_plan in result
                         (needed for 2-stage VSS computation)

        Returns:
            dict with: feasible, physically_feasible, total_cost, slack_cost,
                       slack_total, cost_breakdown, slack_breakdown, status
                       (if return_full: + x_solution, gt_plan)
        """
        P_wind = np.asarray(wind, dtype=float).ravel()[:T]
        P_pv   = np.asarray(solar, dtype=float).ravel()[:T]
        P_elec = np.asarray(elec_load, dtype=float).ravel()[:T]
        P_heat = np.asarray(heat_load, dtype=float).ravel()[:T]
        P_cold = np.asarray(cold_load, dtype=float).ravel()[:T]

        res = self._build_lp(P_wind, P_pv, P_elec, P_heat, P_cold)

        if not res.success:
            return {
                'feasible': False,
                'physically_feasible': False,
                'status': res.message,
                'total_cost': float('inf'),
                'slack_cost': float('inf'),
                'slack_total': float('inf'),
                'cost_breakdown': {},
                'slack_breakdown': {},
            }

        x = res.x

        # Compute cost components from solution
        def sum_var(name):
            return float(x[vidx(name)].sum())

        gas_coef_gt = C_GAS / (ETA_GT * L_CH4)
        gas_coef_gb = C_GAS / (ETA_GB * L_CH4)
        gas_cost = float(np.sum(x[vidx('P_gt')]) * gas_coef_gt +
                         np.sum(x[vidx('Q_gb')]) * gas_coef_gb)
        elec_cost = float(np.sum(x[vidx('P_grid')] * C_BUY))
        om_cost = float(
            C_OM_GT * sum_var('P_gt') + C_OM_WHB * sum_var('Q_whb') +
            C_OM_GB * sum_var('Q_gb') + C_OM_EB * sum_var('Q_eb') +
            C_OM_EC * sum_var('C_ec') + C_OM_AC * sum_var('C_ac') +
            C_OM_BT * (sum_var('P_bt_ch') + sum_var('P_bt_dis')) +
            C_OM_HS * (sum_var('Q_hs_ch') + sum_var('Q_hs_dis')) +
            C_OM_WT * sum_var('P_wind') + C_OM_PV * sum_var('P_pv')
        )
        curtail_cost = float(C_CURTAIL * (
            (P_wind - x[vidx('P_wind')]).sum() +
            (P_pv - x[vidx('P_pv')]).sum()
        ))
        carbon_cost = float(DELTA_CO2 * C_CO2 * sum_var('P_grid'))

        # Slack cost (differentiated by type) and total slack kW
        slack_elec_p = sum_var('S_elec_p')
        slack_elec_n = sum_var('S_elec_n')
        slack_heat_p = sum_var('S_heat_p')
        slack_heat_n = sum_var('S_heat_n')
        slack_cold_p = sum_var('S_cold_p')
        slack_cold_n = sum_var('S_cold_n')

        slack_shed_kw = slack_elec_p + slack_heat_p + slack_cold_p
        slack_excess_kw = slack_elec_n + slack_heat_n + slack_cold_n
        slack_total_kw = slack_shed_kw + slack_excess_kw

        # Per-hour slack (needed for LOLP/EENS reliability metrics)
        def _hourly(name):
            return np.array([float(x[vidx(name, t)]) for t in range(T)])
        per_hour = {
            'shed': _hourly('S_elec_p') + _hourly('S_heat_p') + _hourly('S_cold_p'),
            'shed_elec': _hourly('S_elec_p'),
            'shed_heat': _hourly('S_heat_p'),
            'shed_cold': _hourly('S_cold_p'),
            'excess': _hourly('S_elec_n') + _hourly('S_heat_n') + _hourly('S_cold_n'),
        }

        slack_cost = (C_SHED_ELEC * slack_elec_p + C_SHED_HEAT * slack_heat_p +
                      C_SHED_COLD * slack_cold_p +
                      C_EXCESS * (slack_elec_n + slack_heat_n + slack_cold_n))

        heat_dump_kw = sum_var('Q_dump')
        heat_dump_cost = C_HEAT_DUMP * heat_dump_kw

        # DR costs
        dr_elec = sum_var('DR_elec')
        dr_heat = sum_var('DR_heat')
        dr_cold = sum_var('DR_cold')
        dr_cost = C_DR_ELEC * dr_elec + C_DR_HEAT * dr_heat + C_DR_COLD * dr_cold
        dr_total_kw = dr_elec + dr_heat + dr_cold

        total_cost = (gas_cost + elec_cost + om_cost + curtail_cost + carbon_cost +
                      slack_cost + heat_dump_cost + dr_cost)

        is_physically_feasible = slack_shed_kw < 1.0  # 1 kW — consistent physical threshold

        result = {
            'feasible': True,
            'physically_feasible': is_physically_feasible,
            'status': res.message,
            'total_cost': total_cost,
            'slack_cost': slack_cost,
            'slack_total': slack_total_kw,
            'slack_shed_kw': slack_shed_kw,
            'slack_excess_kw': slack_excess_kw,
            'heat_dump_kw': heat_dump_kw,
            'heat_dump_cost': heat_dump_cost,
            'dr_kw': dr_total_kw,
            'dr_cost': dr_cost,
            'cost_breakdown': {
                'gas': gas_cost,
                'electricity': elec_cost,
                'om': om_cost,
                'curtailment': curtail_cost,
                'carbon': carbon_cost,
                'dr': dr_cost,
                'heat_dump': heat_dump_cost,
                'slack_penalty': slack_cost,
            },
            'slack_breakdown': {
                'elec_p': slack_elec_p,
                'elec_n': slack_elec_n,
                'heat_p': slack_heat_p,
                'heat_n': slack_heat_n,
                'cold_p': slack_cold_p,
                'cold_n': slack_cold_n,
            },
            'per_hour': per_hour,
        }
        if return_full:
            result['x_solution'] = x
            result['gt_plan'] = x[vidx('P_gt')]
        return result

    def solve_batch(self, scenarios_array):
        """Solve for multiple scenarios.

        Args:
            scenarios_array: (N, 24, 5) [wind, solar, elec, heat, cold]

        Returns:
            list of result dicts
        """
        results = []
        s = np.asarray(scenarios_array)
        N = s.shape[0]
        for i in range(N):
            r = self.solve(s[i, :, 0], s[i, :, 1], s[i, :, 2],
                           s[i, :, 3], s[i, :, 4])
            results.append(r)
        return results

    # ------------------------------------------------------------------
    def solve_with_fixed_gt(self, wind, solar, elec_load, heat_load, cold_load,
                            gt_plan):
        """Re-dispatch with GT output fixed to a day-ahead commitment plan.

        Used for 2-stage VSS: Stage 2 (real-time dispatch) of EEV computation.
        Stage 1 (day-ahead) solves EV → gt_plan. Stage 2 re-dispatches each
        scenario with P_gt[t] = gt_plan[t] exactly, measuring the cost of the
        EV commitment being misaligned with the actual scenario.

        Args:
            wind, solar, elec_load, heat_load, cold_load: length-24 arrays (kW)
            gt_plan: length-24 array — GT output from EV (Stage 1) solution

        Returns:
            same dict as solve()
        """
        P_wind = np.asarray(wind, dtype=float).ravel()[:T]
        P_pv   = np.asarray(solar, dtype=float).ravel()[:T]
        P_elec = np.asarray(elec_load, dtype=float).ravel()[:T]
        P_heat = np.asarray(heat_load, dtype=float).ravel()[:T]
        P_cold = np.asarray(cold_load, dtype=float).ravel()[:T]
        P_gt_fixed = np.asarray(gt_plan, dtype=float).ravel()[:T]

        res = self._build_lp(P_wind, P_pv, P_elec, P_heat, P_cold,
                             fixed_gt=P_gt_fixed)

        if not res.success:
            return {
                'feasible': False,
                'physically_feasible': False,
                'status': res.message,
                'total_cost': float('inf'),
                'slack_cost': float('inf'),
                'slack_total': float('inf'),
                'cost_breakdown': {},
                'slack_breakdown': {},
            }

        x = res.x

        def sum_var(name):
            return float(x[vidx(name)].sum())

        gas_coef_gt = C_GAS / (ETA_GT * L_CH4)
        gas_coef_gb = C_GAS / (ETA_GB * L_CH4)
        gas_cost = float(np.sum(x[vidx('P_gt')]) * gas_coef_gt +
                         np.sum(x[vidx('Q_gb')]) * gas_coef_gb)
        elec_cost = float(np.sum(x[vidx('P_grid')] * C_BUY))
        om_cost = float(
            C_OM_GT * sum_var('P_gt') + C_OM_WHB * sum_var('Q_whb') +
            C_OM_GB * sum_var('Q_gb') + C_OM_EB * sum_var('Q_eb') +
            C_OM_EC * sum_var('C_ec') + C_OM_AC * sum_var('C_ac') +
            C_OM_BT * (sum_var('P_bt_ch') + sum_var('P_bt_dis')) +
            C_OM_HS * (sum_var('Q_hs_ch') + sum_var('Q_hs_dis')) +
            C_OM_WT * sum_var('P_wind') + C_OM_PV * sum_var('P_pv')
        )
        curtail_cost = float(C_CURTAIL * (
            (P_wind - x[vidx('P_wind')]).sum() +
            (P_pv - x[vidx('P_pv')]).sum()
        ))
        carbon_cost = float(DELTA_CO2 * C_CO2 * sum_var('P_grid'))

        slack_elec_p = sum_var('S_elec_p')
        slack_elec_n = sum_var('S_elec_n')
        slack_heat_p = sum_var('S_heat_p')
        slack_heat_n = sum_var('S_heat_n')
        slack_cold_p = sum_var('S_cold_p')
        slack_cold_n = sum_var('S_cold_n')

        slack_shed_kw = slack_elec_p + slack_heat_p + slack_cold_p
        slack_excess_kw = slack_elec_n + slack_heat_n + slack_cold_n
        slack_total_kw = slack_shed_kw + slack_excess_kw

        def _hourly(name):
            return np.array([float(x[vidx(name, t)]) for t in range(T)])
        per_hour = {
            'shed': _hourly('S_elec_p') + _hourly('S_heat_p') + _hourly('S_cold_p'),
            'shed_elec': _hourly('S_elec_p'),
            'shed_heat': _hourly('S_heat_p'),
            'shed_cold': _hourly('S_cold_p'),
            'excess': _hourly('S_elec_n') + _hourly('S_heat_n') + _hourly('S_cold_n'),
        }

        slack_cost = (C_SHED_ELEC * slack_elec_p + C_SHED_HEAT * slack_heat_p +
                      C_SHED_COLD * slack_cold_p +
                      C_EXCESS * (slack_elec_n + slack_heat_n + slack_cold_n))

        heat_dump_kw = sum_var('Q_dump')
        heat_dump_cost = C_HEAT_DUMP * heat_dump_kw

        dr_elec = sum_var('DR_elec')
        dr_heat = sum_var('DR_heat')
        dr_cold = sum_var('DR_cold')
        dr_cost = C_DR_ELEC * dr_elec + C_DR_HEAT * dr_heat + C_DR_COLD * dr_cold
        dr_total_kw = dr_elec + dr_heat + dr_cold

        total_cost = (gas_cost + elec_cost + om_cost + curtail_cost + carbon_cost +
                      slack_cost + heat_dump_cost + dr_cost)

        is_physically_feasible = slack_shed_kw < 1.0

        return {
            'feasible': True,
            'physically_feasible': is_physically_feasible,
            'status': res.message,
            'total_cost': total_cost,
            'slack_cost': slack_cost,
            'slack_total': slack_total_kw,
            'slack_shed_kw': slack_shed_kw,
            'slack_excess_kw': slack_excess_kw,
            'heat_dump_kw': heat_dump_kw,
            'heat_dump_cost': heat_dump_cost,
            'dr_kw': dr_total_kw,
            'dr_cost': dr_cost,
            'cost_breakdown': {
                'gas': gas_cost,
                'electricity': elec_cost,
                'om': om_cost,
                'curtailment': curtail_cost,
                'carbon': carbon_cost,
                'dr': dr_cost,
                'heat_dump': heat_dump_cost,
                'slack_penalty': slack_cost,
            },
            'slack_breakdown': {
                'elec_p': slack_elec_p,
                'elec_n': slack_elec_n,
                'heat_p': slack_heat_p,
                'heat_n': slack_heat_n,
                'cold_p': slack_cold_p,
                'cold_n': slack_cold_n,
            },
            'per_hour': per_hour,
        }


# ===========================================================================
# Self-test
# ===========================================================================
if __name__ == '__main__':
    import pandas as pd, os, time

    print("=" * 60)
    print("IES Optimizer Self-Test (scipy.linprog + HiGHS)")
    print("=" * 60)

    opt = IESOptimizer()
    data_path = "./源荷数据集.csv"

    if os.path.exists(data_path):
        df = pd.read_csv(data_path, header=None, encoding='gbk')
        df.columns = ['wind', 'solar', 'electric', 'heat', 'cold']
        full = df.values.astype(np.float64)

        # Test 1: Real day 180
        day180 = full[180*24:181*24]
        t0 = time.time()
        r1 = opt.solve(day180[:, 0], day180[:, 1], day180[:, 2],
                        day180[:, 3], day180[:, 4])
        t1 = time.time()

        print(f"\n[Test 1] Real day 180 (solve time: {(t1-t0)*1000:.1f}ms)")
        print(f"  Status: {r1['status']}")
        print(f"  Feasible: {r1['feasible']}")
        print(f"  Physically feasible: {r1['physically_feasible']}")
        print(f"  Total cost: {r1['total_cost']:.2f} yuan")
        print(f"  Slack cost: {r1['slack_cost']:.2f} yuan")
        print(f"  Cost breakdown:")
        for k, v in r1['cost_breakdown'].items():
            print(f"    {k:>15s}: {v:10.2f}")

        # Test 2: 10x cold (should trigger slack)
        bad = day180.copy()
        bad[:, 4] *= 10
        r2 = opt.solve(bad[:, 0], bad[:, 1], bad[:, 2], bad[:, 3], bad[:, 4])

        print(f"\n[Test 2] 10x cold (physically impossible)")
        print(f"  Feasible: {r2['feasible']}")
        print(f"  Physically feasible: {r2['physically_feasible']}")
        print(f"  Total cost: {r2['total_cost']:.2f} yuan")
        print(f"  Slack cost: {r2['slack_cost']:.2f} yuan")
        print(f"  Slack total: {r2['slack_total']:.2f} kW")
        print(f"  Slack breakdown: {r2['slack_breakdown']}")

        # Test 3: Batch solve 10 real days
        print(f"\n[Test 3] Batch solve 10 real days:")
        t0 = time.time()
        scenarios = np.stack([full[d*24:(d+1)*24] for d in range(10)])
        results = opt.solve_batch(scenarios)
        t1 = time.time()
        feasible = sum(1 for r in results if r['physically_feasible'])
        costs = [r['total_cost'] for r in results if r['feasible']]
        slacks = [r['slack_cost'] for r in results]
        print(f"  Physically feasible: {feasible}/10")
        print(f"  Avg total cost: {np.mean(costs):.2f} yuan")
        print(f"  Avg slack cost: {np.mean(slacks):.2f} yuan")
        print(f"  Max slack cost: {np.max(slacks):.2f} yuan")
        print(f"  Batch time: {(t1-t0)*1000:.1f}ms")

    print(f"\nAll tests complete!")
