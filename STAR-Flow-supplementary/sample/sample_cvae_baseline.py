"""
Sample ConvCVAE baseline for 12 test days × 3 extreme levels.

Uses condition scaling (cond_scale=2.0, analogous to CFG in diffusion) to
amplify the condition signal during sampling.
"""
import torch
import numpy as np
import pandas as pd
import os, warnings
warnings.filterwarnings('ignore')

from cvae_baseline import ConvCVAE
from dataset_energy_v3 import (
    DailyFeatureExtractorV3, EnergyDataset1DV3,
    get_monthly_stratified_split, FEATURE_COLS,
)

DATA_PATH = "./源荷数据集.csv"
RESULTS_DIR = "./results_cvae_baseline"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_SCENARIOS = 100
COND_SCALE = 2.0          # CFG-like condition scaling

LEVELS = [('normal', 0.0), ('high_risk', 0.90), ('extreme', 0.95)]


def main():
    # ---- Load checkpoint ----
    ckpt_path = f"{RESULTS_DIR}/model-best.pt"
    if not os.path.exists(ckpt_path):
        ckpt_path = f"{RESULTS_DIR}/model-final.pt"
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"No checkpoint in {RESULTS_DIR}")

    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    print(f"Loaded: epoch={ckpt.get('epoch', '?')}")

    # ---- Build model ----
    model = ConvCVAE(seq_len=24, channels=5, condition_dim=37, latent_dim=64).to(DEVICE)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    print(f"ConvCVAE loaded ({sum(p.numel() for p in model.parameters()):,} params)")

    # ---- Data pipeline ----
    fe = DailyFeatureExtractorV3(DATA_PATH)
    dataset = EnergyDataset1DV3(DATA_PATH, seq_len=24, normalize=True, feature_extractor=fe)
    _, _, test_days = get_monthly_stratified_split(DATA_PATH)
    test_days = [int(d) for d in test_days]
    print(f"Test: {len(test_days)}d × {len(LEVELS)}lvl × {NUM_SCENARIOS}scn, cond_scale={COND_SCALE}")

    # ---- Sample ----
    os.makedirs(RESULTS_DIR, exist_ok=True)
    all_parts = []

    for day_idx in test_days:
        for lvl_name, lvl_val in LEVELS:
            cond = fe.build_extreme_condition_batch(
                batch_size=NUM_SCENARIOS, day_idx=day_idx,
                extreme_level=lvl_val, noise_std=0.02
            )
            cond_t = torch.FloatTensor(cond).to(DEVICE)

            with torch.no_grad():
                x_hat = model.sample(cond_t, cond_scale=COND_SCALE, device=DEVICE)
                # x_hat: (100, 5, 24)

            # Denormalize
            x_np = x_hat.cpu().numpy()
            arr_list = []
            for i in range(NUM_SCENARIOS):
                tensor_i = torch.FloatTensor(x_np[i])  # (5, 24)
                denormed = dataset.denormalize(tensor_i).numpy().T  # (24, 5)
                arr_list.append(denormed)
            arr_np = np.stack(arr_list, axis=0)  # (100, 24, 5)

            # Nighttime solar clipping
            night_mask = (np.arange(24) < 6) | (np.arange(24) >= 19)
            arr_np[:, night_mask, 1] = 0.0

            # Build CSV rows
            for i in range(NUM_SCENARIOS):
                start_date = pd.Timestamp("2023-01-01") + pd.Timedelta(days=int(day_idx))
                time_idx = pd.date_range(start=start_date, periods=24, freq="h")
                df_tmp = pd.DataFrame(arr_np[i], columns=FEATURE_COLS)
                df_tmp['scenario_id'] = i + 1
                df_tmp['time'] = time_idx
                df_tmp['extreme_level'] = lvl_val
                df_tmp['template_day'] = day_idx
                df_tmp['extreme_label'] = lvl_name
                all_parts.append(df_tmp)

        print(f"  Day {day_idx}: done")

    df_all = pd.concat(all_parts, ignore_index=True)
    cols = ['scenario_id', 'time', 'extreme_level', 'template_day', 'extreme_label'] + FEATURE_COLS
    df_all = df_all[cols]

    out_path = f"{RESULTS_DIR}/generated_scenarios_cvae_all12.csv"
    df_all.to_csv(out_path, index=False, encoding='utf-8-sig')
    print(f"\nSaved {len(df_all)} rows → {out_path}")
    print("ConvCVAE sampling complete!")


if __name__ == '__main__':
    main()
