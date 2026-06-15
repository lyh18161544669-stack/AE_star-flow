"""Batch sample ALL 12 test days (1st of each month) for all 8 models.

Generates 3 levels (normal/high_risk/extreme) × 100 scenarios per day.
Output: generated_scenarios_{model}_all12.csv per model directory.
"""
import sys, os, torch, numpy as np, pandas as pd
import time, warnings
warnings.filterwarnings('ignore')

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEQ_LEN, CHANNELS = 24, 5
DATA_PATH = "./源荷数据集.csv"
FEATURE_COLS = ['wind', 'solar', 'electric', 'heat', 'cold']
NUM_SCENARIOS = 100
COND_SCALE = 2.0

from dataset_energy_v3 import DailyFeatureExtractorV3, EnergyDataset1DV3, get_monthly_stratified_split

fe = DailyFeatureExtractorV3(DATA_PATH)
dataset = EnergyDataset1DV3(DATA_PATH, seq_len=SEQ_LEN, normalize=True, feature_extractor=fe)

_, _, TEST_DAYS = get_monthly_stratified_split(DATA_PATH)
TEST_DAYS = [int(d) for d in TEST_DAYS]

LEVELS = [('normal', 0.0), ('high_risk', 0.90), ('extreme', 0.95)]

print("=" * 70)
print("12-Day Full Coverage Sampling — All 8 Models")
print(f"Test days: {len(TEST_DAYS)} days — {TEST_DAYS}")
print(f"Levels: {[l[0] for l in LEVELS]}")
print(f"Scenarios per level: {NUM_SCENARIOS}")
print(f"Total per model: {len(TEST_DAYS) * len(LEVELS) * NUM_SCENARIOS} scenarios")
print("=" * 70)


def denorm_diff(gen):
    """Diffusion: output (N, C, T) → (N*24, 5) kW array"""
    all_arr = []
    for i in range(NUM_SCENARIOS):
        all_arr.append(dataset.denormalize(gen[i].cpu()).numpy().T)
    return np.concatenate(all_arr, axis=0)


def denorm_v1(gen_np):
    """GAN: output (N, 120) → (N*24, 5) kW array"""
    all_arr = []
    gen_t = gen_np.reshape(NUM_SCENARIOS, CHANNELS, SEQ_LEN)
    for i in range(NUM_SCENARIOS):
        all_arr.append(dataset.denormalize(torch.from_numpy(gen_t[i])).numpy().T)
    return np.concatenate(all_arr, axis=0)


def apply_postprocessing(arr, n_scenarios):
    abs_hours = np.tile(np.arange(SEQ_LEN), n_scenarios)
    night_mask = (abs_hours < 6) | (abs_hours >= 19)
    arr[night_mask, 1] = 0.0
    return arr


def sample_dc_model(diffusion, results_dir, suffix, cond_dim, is_fm=False, cond_pad=0):
    """Sample DualChannel model for all 12 test days, 3 levels each."""
    t0 = time.time()
    all_parts = []
    for day_idx in TEST_DAYS:
        for lvl_name, lvl_val in LEVELS:
            cond = fe.build_extreme_condition_batch(
                NUM_SCENARIOS, day_idx, lvl_val, noise_std=0.02)
            if cond_pad > 0:
                cond = np.pad(cond, ((0, 0), (0, cond_pad)), mode='constant')
            cond_t = torch.FloatTensor(cond).to(DEVICE)
            with torch.no_grad():
                if is_fm:
                    gen = diffusion.sample(batch_size=NUM_SCENARIOS, class_labels=cond_t,
                                           cond_scale=COND_SCALE, num_steps=50)
                else:
                    gen = diffusion.sample(batch_size=NUM_SCENARIOS, class_labels=cond_t,
                                           cond_scale=COND_SCALE)
            arr = denorm_diff(gen)
            arr = apply_postprocessing(arr, NUM_SCENARIOS)
            for i in range(NUM_SCENARIOS):
                start_date = pd.Timestamp("2023-01-01") + pd.Timedelta(days=int(day_idx))
                time_idx = pd.date_range(start=start_date, periods=SEQ_LEN, freq="h")
                df_tmp = pd.DataFrame(arr[i * SEQ_LEN:(i + 1) * SEQ_LEN], columns=FEATURE_COLS)
                df_tmp['scenario_id'] = i + 1
                df_tmp['time'] = time_idx
                df_tmp['extreme_level'] = lvl_val
                df_tmp['template_day'] = day_idx
                df_tmp['extreme_label'] = lvl_name
                all_parts.append(df_tmp)

    df_all = pd.concat(all_parts, ignore_index=True)
    cols = ['scenario_id', 'time', 'extreme_level', 'template_day', 'extreme_label'] + FEATURE_COLS
    df_all = df_all[cols]
    out = f"{results_dir}/generated_scenarios_{suffix}_all12.csv"
    os.makedirs(results_dir, exist_ok=True)
    df_all.to_csv(out, index=False, encoding='utf-8-sig')
    elapsed = time.time() - t0
    print(f"  Saved {len(df_all)} rows → {out}  [{elapsed:.0f}s]")


def load_dc_model(ckpt_path, cond_dim, cross_var, fusion, is_fm):
    from dual_channel_denoiser import DualChannelDenoiser
    denoiser = DualChannelDenoiser(
        seq_len=SEQ_LEN, channels=CHANNELS, condition_dim=cond_dim,
        hidden_dim=256, num_heads=4, num_encoder_layers=2, num_decoder_layers=4,
        mlp_ratio=4, dropout=0.1,
        cross_var_module=cross_var, use_hour_embedding=True,
        fusion_type=fusion,
        time_embed_type='fm' if is_fm else 'sinusoidal',
    )
    if is_fm:
        from flow_matching import ContinuousTimeFlowMatching1D
        diffusion = ContinuousTimeFlowMatching1D(denoiser, seq_length=SEQ_LEN, channels=CHANNELS).to(DEVICE)
    else:
        sys.path.insert(0, 'denoising_diffusion_pytorch')
        from continuous_time_diffusion_1d import ContinuousTimeGaussianDiffusion1D
        diffusion = ContinuousTimeGaussianDiffusion1D(
            denoiser, seq_length=SEQ_LEN, channels=CHANNELS,
            noise_schedule='linear', num_sample_steps=500,
            clip_sample_denoised=True, min_snr_loss_weight=True, min_snr_gamma=5
        ).to(DEVICE)

    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    state = ckpt.get('ema', ckpt.get('model'))
    clean = {}
    for k, v in state.items():
        if k.startswith('ema_model.'):
            clean[k[len('ema_model.'):]] = v
        else:
            clean[k] = v
    for k in diffusion.state_dict():
        if k not in clean:
            clean[k] = diffusion.state_dict()[k]
    diffusion.load_state_dict(clean, strict=False)
    diffusion.eval()
    return diffusion


if __name__ == '__main__':
    total_start = time.time()

    # ===== V1: WGAN-GP =====
    print("\n[1/8] V1 (WGAN-GP)")
    try:
        from wgan_gp_v1 import Generator
        G = Generator(latent_dim=100, condition_dim=37, output_channels=5, output_len=24).to(DEVICE)
        ckpt = torch.load('results_energy_continuous_v1/model-best.pt', map_location=DEVICE, weights_only=False)
        G.load_state_dict(ckpt['G'])
        G.eval()

        t0 = time.time()
        all_parts = []
        for day_idx in TEST_DAYS:
            for lvl_name, lvl_val in LEVELS:
                cond = fe.build_extreme_condition_batch(NUM_SCENARIOS, day_idx, lvl_val, noise_std=0.02)
                cond_t = torch.FloatTensor(cond).to(DEVICE)
                z = torch.randn(NUM_SCENARIOS, 100).to(DEVICE)
                with torch.no_grad():
                    gen_raw = G(z, cond_t).cpu().numpy()
                gen_01 = (gen_raw + 1.0) * 0.5  # Generator outputs [-1,1], convert to [0,1]
                arr = denorm_v1(gen_01)
                arr = apply_postprocessing(arr, NUM_SCENARIOS)
                for i in range(NUM_SCENARIOS):
                    start_date = pd.Timestamp("2023-01-01") + pd.Timedelta(days=int(day_idx))
                    time_idx = pd.date_range(start=start_date, periods=SEQ_LEN, freq="h")
                    df_tmp = pd.DataFrame(arr[i * SEQ_LEN:(i + 1) * SEQ_LEN], columns=FEATURE_COLS)
                    df_tmp['scenario_id'] = i + 1
                    df_tmp['time'] = time_idx
                    df_tmp['extreme_level'] = lvl_val
                    df_tmp['template_day'] = day_idx
                    df_tmp['extreme_label'] = lvl_name
                    all_parts.append(df_tmp)

        df_all = pd.concat(all_parts, ignore_index=True)
        cols = ['scenario_id', 'time', 'extreme_level', 'template_day', 'extreme_label'] + FEATURE_COLS
        df_all = df_all[cols]
        out = "results_energy_continuous_v1/generated_scenarios_v1_all12.csv"
        os.makedirs("results_energy_continuous_v1", exist_ok=True)
        df_all.to_csv(out, index=False, encoding='utf-8-sig')
        print(f"  Saved {len(df_all)} rows → {out}  [{time.time()-t0:.0f}s]")
    except Exception as e:
        print(f"  FAILED V1: {e}")
        import traceback
        traceback.print_exc()

    # ===== V2: ED-DDPM =====
    print("\n[2/8] V2 (ED-DDPM)")
    try:
        sys.path.insert(0, '.')
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
            clip_sample_denoised=True, min_snr_loss_weight=True, min_snr_gamma=5
        ).to(DEVICE)

        ckpt = torch.load('results_energy_continuous_v2/model-best.pt', map_location=DEVICE, weights_only=False)
        state = ckpt.get('ema', ckpt.get('model'))
        clean = {}
        for k, v in state.items():
            if k.startswith('ema_model.'):
                clean[k[len('ema_model.'):]] = v
            else:
                clean[k] = v
        for k in diffusion.state_dict():
            if k not in clean:
                clean[k] = diffusion.state_dict()[k]
        diffusion.load_state_dict(clean, strict=False)
        diffusion.eval()

        t0 = time.time()
        all_parts = []
        for day_idx in TEST_DAYS:
            for lvl_name, lvl_val in LEVELS:
                cond = fe.build_extreme_condition_batch(NUM_SCENARIOS, day_idx, lvl_val, noise_std=0.02)
                cond_t = torch.FloatTensor(cond).to(DEVICE)
                with torch.no_grad():
                    gen = diffusion.sample(batch_size=NUM_SCENARIOS, class_labels=cond_t, cond_scale=COND_SCALE)
                arr = denorm_diff(gen)
                arr = apply_postprocessing(arr, NUM_SCENARIOS)
                for i in range(NUM_SCENARIOS):
                    start_date = pd.Timestamp("2023-01-01") + pd.Timedelta(days=int(day_idx))
                    time_idx = pd.date_range(start=start_date, periods=SEQ_LEN, freq="h")
                    df_tmp = pd.DataFrame(arr[i * SEQ_LEN:(i + 1) * SEQ_LEN], columns=FEATURE_COLS)
                    df_tmp['scenario_id'] = i + 1
                    df_tmp['time'] = time_idx
                    df_tmp['extreme_level'] = lvl_val
                    df_tmp['template_day'] = day_idx
                    df_tmp['extreme_label'] = lvl_name
                    all_parts.append(df_tmp)

        df_all = pd.concat(all_parts, ignore_index=True)
        cols = ['scenario_id', 'time', 'extreme_level', 'template_day', 'extreme_label'] + FEATURE_COLS
        df_all = df_all[cols]
        out = "results_energy_continuous_v2/generated_scenarios_v2_all12.csv"
        os.makedirs("results_energy_continuous_v2", exist_ok=True)
        df_all.to_csv(out, index=False, encoding='utf-8-sig')
        print(f"  Saved {len(df_all)} rows → {out}  [{time.time()-t0:.0f}s]")
    except Exception as e:
        print(f"  FAILED V2: {e}")
        import traceback
        traceback.print_exc()

    # ===== V2.5 - V4: DualChannel models =====
    dc_models = [
        ('V2.5 (DC-DDPM)', 'results_energy_continuous_v2.5/model-best.pt',
         'results_energy_continuous_v2.5', 'v2.5', 37, 'star', 'additive', False),
        ('V3 (DC-CAF)', 'results_energy_continuous_v3/model-best.pt',
         'results_energy_continuous_v3', 'v3', 37, 'star', 'cross_attn_gate', False),
        ('V3-FM (DC-CAF-FM)', 'results_energy_continuous_v3_fm/model-best.pt',
         'results_energy_continuous_v3_fm', 'v3_fm', 37, 'star', 'cross_attn_gate', True),
        ('V3-IESGAT (DC-CAF-GAT)', 'results_energy_continuous_v3_iesgat/model-best.pt',
         'results_energy_continuous_v3_iesgat', 'v3_iesgat', 37, 'ies_gat', 'cross_attn_gate', False),
        ('V3-FM-IESGAT (DC-CAF-FM-GAT)', 'results_energy_continuous_v3_fm_iesgat/model-best.pt',
         'results_energy_continuous_v3_fm_iesgat', 'v3_fm_iesgat', 37, 'ies_gat', 'cross_attn_gate', True),
        ('V4 (DC-GAT)', 'results_energy_continuous_v4/model-best.pt',
         'results_energy_continuous_v4', 'v4', 101, 'star', 'additive', False),
    ]

    for idx, (label, ckpt_path, res_dir, suffix, cond_dim, cv, fusion, is_fm) in enumerate(dc_models):
        print(f"\n[{idx+3}/8] {label}")
        try:
            diffusion = load_dc_model(ckpt_path, cond_dim, cv, fusion, is_fm)
            cond_pad = 64 if 'V4' in label else 0
            sample_dc_model(diffusion, res_dir, suffix, cond_dim, is_fm=is_fm, cond_pad=cond_pad)
        except Exception as e:
            print(f"  FAILED {label}: {e}")
            import traceback
            traceback.print_exc()

    elapsed = time.time() - total_start
    print("\n" + "=" * 70)
    print(f"All done! Total time: {elapsed/60:.1f} min ({elapsed:.0f}s)")
    print(f"12 test days × 3 levels × 100 scenarios × 8 models")
    print("=" * 70)
