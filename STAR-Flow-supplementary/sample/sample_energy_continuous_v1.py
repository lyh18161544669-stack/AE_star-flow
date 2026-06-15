"""
V1 WGAN-GP Sampling: Conditional energy scenario generation (multi-level).

Generates 100 scenarios per extreme level (normal/moderate/extreme/severe)
conditioned on test day 181 (Jul 1) for fair comparison with V2.5+.
"""

import torch
import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler

from wgan_gp_v1 import WGAN_GP, Generator
from dataset_energy_v3 import (DailyFeatureExtractorV3, EnergyDataset1DV3,
                                FEATURE_COLS, load_energy_data)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEQ_LEN = 24
CHANNELS = 5
CONDITION_DIM = 37
LATENT_DIM = 100
NUM_SCENARIOS = 100
RESULTS_FOLDER = "./results_energy_continuous_v1"
MODEL_PATH = f"{RESULTS_FOLDER}/model-best.pt"
DATA_PATH = "./源荷数据集.csv"
TEMPLATE_DAY = 181
OUTPUT_PATH = f"{RESULTS_FOLDER}/generated_scenarios_v1.csv"
MULTI_LEVEL_COMPARE = True


def load_model(model_path):
    if not os.path.exists(model_path):
        available = sorted([
            f for f in os.listdir(RESULTS_FOLDER)
            if f.startswith('model-') and f.endswith('.pt') and f != 'model-best.pt'
        ], key=lambda x: int(x.replace('model-', '').replace('.pt', '')))
        if available:
            model_path = os.path.join(RESULTS_FOLDER, available[-1])
            print(f"best model not found, fallback: {model_path}")
        else:
            raise FileNotFoundError(f"No checkpoint in {RESULTS_FOLDER}/")

    checkpoint = torch.load(model_path, map_location=DEVICE, weights_only=False)
    G = Generator(LATENT_DIM, CONDITION_DIM, CHANNELS, SEQ_LEN).to(DEVICE)
    G.load_state_dict(checkpoint['G'])
    G.eval()

    if 'data_min' in checkpoint:
        scaler = MinMaxScaler()
        scaler.data_min_ = checkpoint['data_min']
        scaler.data_max_ = checkpoint['data_max']
        scaler.data_range_ = checkpoint.get('data_range_',
                                             scaler.data_max_ - scaler.data_min_)
        scaler.scale_ = checkpoint.get('scale_', 1.0 / scaler.data_range_)
        scaler.min_ = -scaler.data_min_ * scaler.scale_
        scaler.n_features_in_ = 5
        scaler.n_samples_seen_ = 8760
    else:
        scaler = None

    return G, scaler, checkpoint.get('step', 'unknown')


def apply_postprocessing(scenarios_array, dataset, num_scenarios):
    absolute_hours = np.tile(np.arange(SEQ_LEN), num_scenarios)
    night_mask = (absolute_hours < 6) | (absolute_hours >= 19)
    night_before = scenarios_array[night_mask, 1].copy()
    scenarios_array[night_mask, 1] = 0.0
    n_clipped = int((night_before > 0.01).sum())
    if n_clipped > 0:
        print(f"  P0 solar night clip: {n_clipped} values (max_before={night_before.max():.1f})")
    return scenarios_array, n_clipped


def generate_at_level(G, dataset, fe, extreme_level, per_variable, num_scenarios, template_day):
    cond_37 = fe.build_extreme_condition_batch(
        batch_size=num_scenarios, day_idx=template_day,
        extreme_level=extreme_level, per_variable=per_variable, noise_std=0.02
    )
    cond_37 = torch.FloatTensor(cond_37).to(DEVICE)

    with torch.no_grad():
        z = torch.randn(num_scenarios, LATENT_DIM, device=DEVICE)
        fake_norm = G(z, cond_37)
        fake_01 = (fake_norm + 1.0) * 0.5
        generated = fake_01.cpu()

    all_scenarios = []
    for i in range(num_scenarios):
        denormed = dataset.denormalize(generated[i]).numpy().T
        all_scenarios.append(denormed)
    scenarios_array = np.concatenate(all_scenarios, axis=0)
    scenarios_array, _ = apply_postprocessing(scenarios_array, dataset, num_scenarios)
    return scenarios_array


def generate_multi_level(G, dataset, fe, template_day=TEMPLATE_DAY):
    levels = {
        'normal': (0.0, None),
        'high_risk': (0.90, None),
        'extreme': (0.95, None),
    }
    results = {}
    for label, (level, pv) in levels.items():
        print(f"\n[{label.upper()}] extreme_level={level}")
        results[label] = generate_at_level(
            G, dataset, fe, level, pv, NUM_SCENARIOS, template_day
        )
    return results


def save_results(scenarios_array, num_scenarios, output_path, extreme_level,
                  per_variable, fe, template_day, level_label=None):
    df_list = []
    for i in range(num_scenarios):
        start_date = pd.Timestamp("2023-01-01") + pd.Timedelta(days=int(template_day))
        time_index = pd.date_range(start=start_date, periods=SEQ_LEN, freq="h")
        df_temp = pd.DataFrame(
            scenarios_array[i * SEQ_LEN: (i + 1) * SEQ_LEN],
            columns=FEATURE_COLS
        )
        df_temp['scenario_id'] = i + 1
        df_temp['time'] = time_index
        df_temp['extreme_level'] = extreme_level
        df_temp['template_day'] = template_day
        if level_label:
            df_temp['extreme_label'] = level_label
        df_list.append(df_temp)

    df_final = pd.concat(df_list, ignore_index=True)
    cols = ['scenario_id', 'time', 'extreme_level', 'template_day']
    if level_label:
        cols.append('extreme_label')
    df_final = df_final[cols + FEATURE_COLS]
    if output_path:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        df_final.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"\nSaved to: {output_path}")
    return df_final


def print_comparison(multi_results, dataset):
    orig = dataset.data
    print(f"\n{'='*70}")
    print(f"Multi-Level Comparison (template_day={TEMPLATE_DAY})")
    print(f"{'='*70}")
    header = f"{'Variable':>10s}"
    for label in multi_results:
        header += f" {label:>10s}"
    header += f" {'Original':>10s}"
    print(header)
    print('-' * 70)
    for c, name in enumerate(FEATURE_COLS):
        row = f"{name:>10s}"
        for label in multi_results:
            vals = multi_results[label][:, c]
            row += f" {vals.mean():10.1f}"
        row += f" {orig[:, c].mean():10.1f}"
        print(row)

    print(f"\n--- P99 ---")
    header2 = f"{'Variable':>10s}"
    for label in multi_results:
        header2 += f" {label:>10s}"
    header2 += f" {'Original':>10s}"
    print(header2)
    print('-' * 70)
    for c, name in enumerate(FEATURE_COLS):
        row = f"{name:>10s}"
        for label in multi_results:
            vals = multi_results[label][:, c]
            row += f" {np.percentile(vals, 99):10.1f}"
        row += f" {np.percentile(orig[:, c], 99):10.1f}"
        print(row)


if __name__ == '__main__':
    print("=" * 60)
    print("V1 WGAN-GP Sampling — Multi-Level EVT Generation")
    print("=" * 60)

    print("\n[1/3] Loading model & data...")
    G, ckpt_scaler, step = load_model(MODEL_PATH)
    print(f"V1 WGAN-GP loaded: step={step}")

    fe = DailyFeatureExtractorV3(DATA_PATH)
    fe.print_summary()
    if ckpt_scaler is not None:
        dataset = EnergyDataset1DV3(DATA_PATH, seq_len=SEQ_LEN, normalize=True,
                                     feature_extractor=fe)
        dataset.scaler = ckpt_scaler
        dataset.data_normalized = ckpt_scaler.transform(dataset.data)
    else:
        dataset = EnergyDataset1DV3(DATA_PATH, seq_len=SEQ_LEN, normalize=True,
                                     feature_extractor=fe)

    print(f"\n[2/3] Config: day={TEMPLATE_DAY}, dim={CONDITION_DIM}")
    os.makedirs(RESULTS_FOLDER, exist_ok=True)

    if MULTI_LEVEL_COMPARE:
        print(f"\n[3/3] Multi-level generation (4 levels x {NUM_SCENARIOS} scenarios)...")
        multi_results = generate_multi_level(G, dataset, fe, TEMPLATE_DAY)

        level_info = {
            'normal': (0.0, 'normal'),
            'high_risk': (0.90, 'high_risk'),
            'extreme': (0.95, 'extreme'),
        }
        df_list_all = []
        for label, arr in multi_results.items():
            lvl, lbl = level_info.get(label, (0.95, label))
            n_days = len(arr) // 24
            df_lvl = save_results(arr, n_days, None, lvl, None, fe, TEMPLATE_DAY,
                                  level_label=lbl)
            df_list_all.append(df_lvl)

        combined = pd.concat(df_list_all, ignore_index=True)
        os.makedirs(os.path.dirname(OUTPUT_PATH) or ".", exist_ok=True)
        combined.to_csv(OUTPUT_PATH, index=False, encoding='utf-8-sig')
        print(f"\nSaved {sum(len(multi_results[l])//24 for l in multi_results)} "
              f"scenarios to: {OUTPUT_PATH}")
        print_comparison(multi_results, dataset)
    else:
        print(f"\n[3/3] Generating {NUM_SCENARIOS} scenarios (day {TEMPLATE_DAY})...")
        cond_37 = torch.FloatTensor(
            fe.get_condition(TEMPLATE_DAY)
        ).unsqueeze(0).repeat(NUM_SCENARIOS, 1).to(DEVICE)

        with torch.no_grad():
            z = torch.randn(NUM_SCENARIOS, LATENT_DIM, device=DEVICE)
            fake_norm = G(z, cond_37)
            fake_01 = (fake_norm + 1.0) * 0.5
            generated = fake_01.cpu()

        all_scenarios = []
        for i in range(NUM_SCENARIOS):
            denormed = dataset.denormalize(generated[i]).numpy().T
            all_scenarios.append(denormed)
        scenarios_array = np.concatenate(all_scenarios, axis=0)
        scenarios_array, n_clipped = apply_postprocessing(scenarios_array, dataset, NUM_SCENARIOS)
        df = save_results(scenarios_array, NUM_SCENARIOS, OUTPUT_PATH, 0.0, None, fe, TEMPLATE_DAY)

    sample_save_path = f"{RESULTS_FOLDER}/final_samples.pt"
    if MULTI_LEVEL_COMPARE:
        torch.save({label: torch.FloatTensor(arr).reshape(-1, 5, 24)
                    for label, arr in multi_results.items()}, sample_save_path)
    print(f"\nSamples saved to: {sample_save_path}")
    print(f"\n{'='*60}")
    print("V1 WGAN-GP sampling complete!")
    print(f"{'='*60}")
