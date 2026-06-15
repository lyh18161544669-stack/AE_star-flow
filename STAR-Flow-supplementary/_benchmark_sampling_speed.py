"""
Benchmark: wall-clock sampling time for DDPM vs FM models.

Measures per-scenario time and total time for full 12-day x 3-level evaluation.
Key comparison: V3 (DDPM, 500 steps) vs V3-FM (FM, 50/100 steps)
— same DualChannel architecture, different generative framework.

Also sweeps FM num_steps to measure CRPS vs NFE tradeoff (Pareto frontier).

Output: benchmark_results.json + printed summary table.
"""
import time, json, os, sys, warnings
import numpy as np, torch, pandas as pd
warnings.filterwarnings('ignore')

from sklearn.preprocessing import MinMaxScaler
from dataset_energy_v3 import DailyFeatureExtractorV3, EnergyDataset1DV3, FEATURE_COLS, load_energy_data, get_monthly_stratified_split
from dual_channel_denoiser import DualChannelDenoiser
from evaluation_metrics import compute_crps_all

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_PATH = "./源荷数据集.csv"
SEQ_LEN, CHANNELS = 24, 5
NUM_SCENARIOS = 100

# Only measure on 3 representative days to save time
# Results scale linearly, so per-scenario time is the key metric
BENCHMARK_DAYS = {
    'winter_jan': 0,
    'shoulder_may': 120,
    'summer_jul': 181,
}
LEVELS_CONFIG = [('normal', 0.0), ('high_risk', 0.90), ('extreme', 0.95)]


def load_v3_model(is_fm):
    """Load V3 or V3-FM checkpoint. Same architecture, different framework."""
    from denoising_diffusion_pytorch.continuous_time_diffusion_1d import ContinuousTimeGaussianDiffusion1D
    from flow_matching import ContinuousTimeFlowMatching1D

    denoiser = DualChannelDenoiser(
        seq_len=SEQ_LEN, channels=CHANNELS, condition_dim=37,
        hidden_dim=256, num_heads=4, num_encoder_layers=2, num_decoder_layers=4,
        mlp_ratio=4, dropout=0.1,
        cross_var_module='star', fusion_type='cross_attn_gate',
        time_embed_type='fm' if is_fm else 'sinusoidal',
        use_hour_embedding=True,
    )

    if is_fm:
        ckpt_path = 'results_energy_continuous_v3_fm/model-best.pt'
        diffusion = ContinuousTimeFlowMatching1D(denoiser, seq_length=SEQ_LEN, channels=CHANNELS).to(DEVICE)
    else:
        ckpt_path = 'results_energy_continuous_v3/model-best.pt'
        diffusion = ContinuousTimeGaussianDiffusion1D(
            denoiser, seq_length=SEQ_LEN, channels=CHANNELS,
            noise_schedule='linear', num_sample_steps=500,
            clip_sample_denoised=True, min_snr_loss_weight=True, min_snr_gamma=5,
        ).to(DEVICE)

    checkpoint = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    state = checkpoint.get('ema', checkpoint.get('model', {}))
    clean = {}
    for k, v in state.items():
        clean[k[len('ema_model.'):] if k.startswith('ema_model.') else k] = v
    for k in diffusion.state_dict():
        if k not in clean:
            clean[k] = diffusion.state_dict()[k]
    clean = {k: v for k, v in clean.items() if k in diffusion.state_dict()}
    diffusion.load_state_dict(clean, strict=False)
    diffusion.eval()
    return diffusion, denoiser


def load_v2_model():
    """Load V2 (ED-DDPM, Karras UNet baseline)."""
    from denoising_diffusion_pytorch.karras_unet_1d import KarrasUnet1D
    from denoising_diffusion_pytorch.continuous_time_diffusion_1d import ContinuousTimeGaussianDiffusion1D

    model = KarrasUnet1D(
        seq_len=SEQ_LEN, dim=192, dim_max=768, channels=CHANNELS, condition_dim=37,
        num_downsamples=2, num_blocks_per_stage=2, attn_res=(12, 6),
        fourier_dim=16, attn_dim_head=64, dropout=0.1,
        self_condition=False, use_positional_encoding=True, use_hour_embedding=True,
    ).to(DEVICE)
    diffusion = ContinuousTimeGaussianDiffusion1D(
        model, seq_length=SEQ_LEN, channels=CHANNELS,
        noise_schedule='linear', num_sample_steps=500,
        clip_sample_denoised=True, min_snr_loss_weight=True, min_snr_gamma=5,
    ).to(DEVICE)

    ckpt_path = 'results_energy_continuous_v2/model-best.pt'
    checkpoint = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    state = checkpoint.get('ema', checkpoint.get('model', {}))
    clean = {}
    for k, v in state.items():
        clean[k[len('ema_model.'):] if k.startswith('ema_model.') else k] = v
    for k in diffusion.state_dict():
        if k not in clean:
            clean[k] = diffusion.state_dict()[k]
    diffusion.load_state_dict(clean, strict=False)
    diffusion.eval()
    return diffusion


def sample_day(diffusion, dataset, fe, day_idx, level, is_fm, num_steps, cond_pad=0):
    """Sample 100 scenarios for one day/level. Returns (scenarios_raw, wall_time)."""
    cond = fe.build_extreme_condition_batch(
        batch_size=NUM_SCENARIOS, day_idx=day_idx,
        extreme_level=float(level), noise_std=0.02
    )
    if cond_pad > 0:
        cond = np.pad(cond, ((0, 0), (0, cond_pad)), mode='constant')
    cond_t = torch.FloatTensor(cond).to(DEVICE)

    torch.cuda.synchronize() if torch.cuda.is_available() else None
    t0 = time.perf_counter()

    with torch.no_grad():
        if is_fm:
            gen = diffusion.sample(batch_size=NUM_SCENARIOS, class_labels=cond_t,
                                   cond_scale=2.0, num_steps=num_steps, show_progress=False)
        else:
            gen = diffusion.sample(batch_size=NUM_SCENARIOS, class_labels=cond_t,
                                   cond_scale=2.0)

    torch.cuda.synchronize() if torch.cuda.is_available() else None
    elapsed = time.perf_counter() - t0

    arr_list = []
    for i in range(NUM_SCENARIOS):
        arr_list.append(dataset.denormalize(gen[i].cpu()).numpy().T)
    arr = np.concatenate(arr_list, axis=0)
    abs_hours = np.tile(np.arange(24), NUM_SCENARIOS)
    arr[(abs_hours < 6) | (abs_hours >= 19), 1] = 0.0
    return arr.reshape(NUM_SCENARIOS, 24, 5), elapsed


def compute_crps_for_scenarios(scenarios_norm, real_day_ct):
    """Quick CRPS evaluation."""
    gen_nct = scenarios_norm.transpose(0, 2, 1)
    crps = compute_crps_all(gen_nct, real_day_ct)
    return crps['crps_sum']


def main():
    print("=" * 70)
    print("SAMPLING SPEED BENCHMARK + NFE-QUALITY SWEEP")
    print(f"Device: {DEVICE}")
    print("=" * 70)

    # ---- Load real data ----
    raw, _ = load_energy_data(DATA_PATH)
    full = raw.astype(np.float32)
    scaler = MinMaxScaler(); scaler.fit(full)
    norm = scaler.transform(full)
    days_norm = norm.reshape(365, 24, 5)
    days_nct = days_norm.transpose(0, 2, 1)

    fe = DailyFeatureExtractorV3(DATA_PATH)
    dataset = EnergyDataset1DV3(DATA_PATH, seq_len=SEQ_LEN, normalize=True, feature_extractor=fe)

    results = {}

    # =====================================================================
    # Part 1: Wall-Clock Timing (3 models, 3 days, 3 levels)
    # =====================================================================
    print("\n" + "=" * 70)
    print("PART 1: WALL-CLOCK SAMPLING TIME")
    print("=" * 70)

    timing_models = [
        ('V2 (ED-DDPM, KarrasUNet)', 'v2', False, 500),
        ('V3 (DC-CAF, DDPM)', 'v3', False, 500),
        ('V3-FM (STAR-Flow, FM)', 'v3_fm', True, 50),
    ]

    for model_name, model_key, is_fm, default_steps in timing_models:
        print(f"\n[{model_name}] Loading...")
        try:
            if model_key == 'v2':
                diffusion = load_v2_model()
            else:
                diffusion, _ = load_v3_model(is_fm)
        except Exception as e:
            print(f"  SKIP: {e}")
            continue

        timing_records = []
        total_elapsed = 0.0
        total_runs = 0

        for day_name, day_idx in BENCHMARK_DAYS.items():
            for lvl_name, lvl_val in LEVELS_CONFIG:
                print(f"  Day {day_idx} ({day_name}), Level={lvl_name} ({lvl_val})...", end=' ', flush=True)
                try:
                    scenarios_raw, elapsed = sample_day(
                        diffusion, dataset, fe, day_idx, lvl_val, is_fm,
                        num_steps=default_steps
                    )
                    timing_records.append({
                        'model': model_name, 'day': day_name, 'day_idx': day_idx,
                        'level': lvl_name, 'level_val': lvl_val,
                        'num_steps': default_steps, 'wall_time_s': elapsed,
                        'time_per_scenario_ms': elapsed / NUM_SCENARIOS * 1000,
                    })
                    total_elapsed += elapsed
                    total_runs += 1
                    print(f"{elapsed:.1f}s ({elapsed/NUM_SCENARIOS*1000:.1f}ms/scenario)")
                except Exception as e:
                    print(f"FAILED: {e}")

        avg_per_scenario = total_elapsed / (total_runs * NUM_SCENARIOS) if total_runs > 0 else 0
        # Extrapolate to full 12-day eval
        full_eval_est = avg_per_scenario * 12 * 3 * NUM_SCENARIOS

        print(f"  Summary: avg {avg_per_scenario*1000:.1f}ms/scenario, "
              f"est. full 12d eval: {full_eval_est/60:.1f}min")

        results[model_name] = {
            'type': 'wall_clock',
            'records': timing_records,
            'avg_ms_per_scenario': avg_per_scenario * 1000,
            'est_full_eval_minutes': full_eval_est / 60,
            'total_benchmark_s': total_elapsed,
        }

    # =====================================================================
    # Part 2: NFE-Quality Sweep (V3-FM only, 1 day, all levels)
    # =====================================================================
    print("\n" + "=" * 70)
    print("PART 2: NFE-QUALITY TRADEOFF (V3-FM, num_steps sweep)")
    print("=" * 70)

    try:
        diffusion_fm, _ = load_v3_model(is_fm=True)
    except Exception as e:
        print(f"V3-FM load failed: {e}")
        diffusion_fm = None

    if diffusion_fm is not None:
        nfe_sweep = [10, 25, 50, 100, 200, 500]
        sweep_results = []

        # Use day 181 (summer) as representative day
        sweep_day = 181
        sweep_levels = [('normal', 0.0), ('extreme', 0.95)]

        for num_steps in nfe_sweep:
            print(f"\n  num_steps={num_steps}...", flush=True)
            crps_values = {}
            time_values = {}

            for lvl_name, lvl_val in sweep_levels:
                try:
                    scenarios_raw, elapsed = sample_day(
                        diffusion_fm, dataset, fe, sweep_day, lvl_val,
                        is_fm=True, num_steps=num_steps
                    )
                    scenarios_norm = scaler.transform(scenarios_raw.reshape(-1, 5)).reshape(-1, 24, 5)
                    crps = compute_crps_for_scenarios(scenarios_norm, days_nct[sweep_day])
                    crps_values[lvl_name] = crps
                    time_values[lvl_name] = elapsed
                    print(f"    {lvl_name}: CRPS-sum={crps:.4f}, time={elapsed:.1f}s")
                except Exception as e:
                    print(f"    {lvl_name} FAILED: {e}")
                    crps_values[lvl_name] = float('nan')
                    time_values[lvl_name] = float('nan')

            sweep_results.append({
                'num_steps': num_steps,
                'nfe': num_steps,  # NFE = num_steps for ODE
                'crps_normal': crps_values.get('normal', float('nan')),
                'crps_extreme': crps_values.get('extreme', float('nan')),
                'time_normal_s': time_values.get('normal', float('nan')),
                'time_extreme_s': time_values.get('extreme', float('nan')),
            })

            print(f"  => NFE={num_steps}: CRPS-normal={crps_values.get('normal',float('nan')):.4f}, "
                  f"CRPS-extreme={crps_values.get('extreme',float('nan')):.4f}")

        results['V3-FM NFE Sweep'] = {
            'type': 'nfe_quality',
            'sweep_day': sweep_day,
            'results': sweep_results,
        }

    # =====================================================================
    # Part 3: Also test DDPM with reduced steps (V3, same architecture)
    # =====================================================================
    print("\n" + "=" * 70)
    print("PART 3: DDPM NFE-QUALITY (V3, num_sample_steps sweep)")
    print("=" * 70)

    try:
        diffusion_ddpm, _ = load_v3_model(is_fm=False)
    except Exception as e:
        print(f"V3 load failed: {e}")
        diffusion_ddpm = None

    if diffusion_ddpm is not None:
        d_steps = [50, 100, 250, 500]
        ddpm_sweep = []

        for n_steps in d_steps:
            diffusion_ddpm.num_sample_steps = n_steps  # patch DDPM steps
            print(f"\n  num_sample_steps={n_steps}...", flush=True)
            crps_values = {}
            time_values = {}

            for lvl_name, lvl_val in [('normal', 0.0), ('extreme', 0.95)]:
                try:
                    scenarios_raw, elapsed = sample_day(
                        diffusion_ddpm, dataset, fe, sweep_day, lvl_val,
                        is_fm=False, num_steps=n_steps
                    )
                    scenarios_norm = scaler.transform(scenarios_raw.reshape(-1, 5)).reshape(-1, 24, 5)
                    crps = compute_crps_for_scenarios(scenarios_norm, days_nct[sweep_day])
                    crps_values[lvl_name] = crps
                    time_values[lvl_name] = elapsed
                    print(f"    {lvl_name}: CRPS-sum={crps:.4f}, time={elapsed:.1f}s")
                except Exception as e:
                    print(f"    {lvl_name} FAILED: {e}")
                    crps_values[lvl_name] = float('nan')
                    time_values[lvl_name] = float('nan')

            ddpm_sweep.append({
                'num_sample_steps': n_steps,
                'nfe': n_steps,
                'crps_normal': crps_values.get('normal', float('nan')),
                'crps_extreme': crps_values.get('extreme', float('nan')),
                'time_normal_s': time_values.get('normal', float('nan')),
                'time_extreme_s': time_values.get('extreme', float('nan')),
            })

        results['V3 DDPM NFE Sweep'] = {
            'type': 'nfe_quality',
            'sweep_day': sweep_day,
            'results': ddpm_sweep,
        }

    # =====================================================================
    # Save
    # =====================================================================
    def to_json(obj):
        if isinstance(obj, dict):
            return {k: to_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [to_json(x) for x in obj]
        elif isinstance(obj, (np.floating, np.integer)):
            return float(obj) if not np.isnan(float(obj)) else None
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    output_path = "benchmark_results.json"
    with open(output_path, 'w') as f:
        json.dump(to_json(results), f, indent=2)
    print(f"\nSaved: {output_path}")

    # =====================================================================
    # Summary Table
    # =====================================================================
    print("\n" + "=" * 70)
    print("SUMMARY: Wall-Clock Sampling Time")
    print("=" * 70)
    print(f"{'Model':<30s} {'Steps':>6s} {'ms/scene':>10s} {'3d×3l (s)':>12s} {'Est.12d (min)':>14s}")
    print("-" * 72)
    for model_name, data in results.items():
        if data.get('type') == 'wall_clock':
            recs = data['records']
            if recs:
                steps = recs[0]['num_steps']
                avg_ms = data['avg_ms_per_scenario']
                total_s = data['total_benchmark_s']
                est_min = data['est_full_eval_minutes']
                print(f"{model_name:<30s} {steps:>6d} {avg_ms:>10.1f} {total_s:>12.1f} {est_min:>14.1f}")

    print(f"\n{'=' * 70}")
    print("SUMMARY: NFE-Quality Tradeoff (Day 181, Normal level)")
    print("=" * 70)
    print(f"{'Model':<25s} {'NFE':>6s} {'CRPS':>8s} {'Time(s)':>8s}")
    print("-" * 48)

    for key in ['V3-FM NFE Sweep', 'V3 DDPM NFE Sweep']:
        if key in results:
            model_label = 'V3-FM (FM)' if 'FM' in key else 'V3 (DDPM)'
            for r in results[key]['results']:
                print(f"{model_label:<25s} {r['nfe']:>6d} {r['crps_normal']:>8.4f} {r['time_normal_s']:>8.1f}")

    print(f"\nDone. Full results in {output_path}")


if __name__ == '__main__':
    main()
