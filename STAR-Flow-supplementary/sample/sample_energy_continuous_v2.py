"""
V2 采样: KarrasUnet1D + 37维EVT条件 + CFG (ED-DDPM)

关键特性:
  - Karras UNet 1D架构 (CNN backbone, 非Transformer)
  - 37维条件 (DailyFeatureExtractorV3, 与V3+共享EVT合成)
  - 支持3级极端程度 (normal/high_risk/extreme)
  - CFG cond_scale=2.0, hour embedding, positional encoding
"""
import torch
import pandas as pd
import numpy as np
import os

from denoising_diffusion_pytorch.karras_unet_1d import KarrasUnet1D
from denoising_diffusion_pytorch.continuous_time_diffusion_1d import (
    ContinuousTimeGaussianDiffusion1D
)
from dataset_energy_v3 import DailyFeatureExtractorV3, EnergyDataset1DV3, FEATURE_COLS

# ================= 配置 =================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEQ_LEN = 24
CHANNELS = 5
CONDITION_DIM = 37
NUM_SCENARIOS = 100
RESULTS_FOLDER = "./results_energy_continuous_v2"
MODEL_PATH = f"{RESULTS_FOLDER}/model-best.pt"
DATA_PATH = "./源荷数据集.csv"

TEMPLATE_DAY = 181
COND_SCALE = 2.0
MULTI_LEVEL_COMPARE = True
OUTPUT_PATH = f"{RESULTS_FOLDER}/generated_scenarios_v2.csv"

# V2 架构参数 (与训练完全一致)
UNET_DIM = 192
UNET_DIM_MAX = 768
UNET_FOURIER_DIM = 16
UNET_ATTN_DIM_HEAD = 64
UNET_DROPOUT = 0.1


def load_model(model_path):
    if not os.path.exists(model_path):
        available = sorted([
            f for f in os.listdir(RESULTS_FOLDER)
            if f.startswith('model-') and f.endswith('.pt') and f != 'model-best.pt'
        ], key=lambda x: int(x.replace('model-', '').replace('.pt', '')))
        if available:
            model_path = os.path.join(RESULTS_FOLDER, available[-1])
            print(f"best model未找到, 回退至: {model_path}")
        else:
            raise FileNotFoundError(f"未找到任何checkpoint: {RESULTS_FOLDER}/")

    model = KarrasUnet1D(
        seq_len=SEQ_LEN,
        dim=UNET_DIM,
        dim_max=UNET_DIM_MAX,
        channels=CHANNELS,
        condition_dim=CONDITION_DIM,
        num_downsamples=2,
        num_blocks_per_stage=2,
        attn_res=(12, 6),
        fourier_dim=UNET_FOURIER_DIM,
        attn_dim_head=UNET_ATTN_DIM_HEAD,
        dropout=UNET_DROPOUT,
        self_condition=False,
        use_positional_encoding=True,
        use_hour_embedding=True,
    )

    diffusion = ContinuousTimeGaussianDiffusion1D(
        model, seq_length=SEQ_LEN, channels=CHANNELS,
        noise_schedule='linear', num_sample_steps=500,
        clip_sample_denoised=True, min_snr_loss_weight=True, min_snr_gamma=5
    ).to(DEVICE)

    checkpoint = torch.load(model_path, map_location=DEVICE, weights_only=False)

    if 'ema' in checkpoint:
        clean_state = {}
        for k, v in checkpoint['ema'].items():
            if k.startswith('ema_model.'):
                clean_state[k[len('ema_model.'):]] = v
        # CFG backward compat
        if not any('null_condition_emb' in k for k in clean_state):
            print("  Old checkpoint (no CFG): initializing null_condition_emb")
            for k in diffusion.state_dict():
                if 'null_condition_emb' in k:
                    clean_state[k] = diffusion.state_dict()[k]
        # Hour embedding backward compat (pad input channels)
        for k in list(clean_state.keys()):
            if 'input_block.weight' in k and k in diffusion.state_dict():
                ckpt_w = clean_state[k]
                model_w = diffusion.state_dict()[k]
                if ckpt_w.shape[1] < model_w.shape[1]:
                    print(f"  Old checkpoint (no hour_emb): padding {k} "
                          f"{ckpt_w.shape[1]}→{model_w.shape[1]} channels")
                    pad = torch.zeros(model_w.shape[0],
                                      model_w.shape[1] - ckpt_w.shape[1],
                                      model_w.shape[2])
                    clean_state[k] = torch.cat([ckpt_w, pad], dim=1)
            if 'temporal_proj.weight' in k and k in diffusion.state_dict():
                ckpt_w = clean_state[k]
                model_w = diffusion.state_dict()[k]
                if ckpt_w.shape[1] < model_w.shape[1]:
                    print(f"  Old checkpoint (no hour_emb): padding {k} "
                          f"{ckpt_w.shape[1]}→{model_w.shape[1]} channels")
                    pad = torch.zeros(model_w.shape[0],
                                      model_w.shape[1] - ckpt_w.shape[1])
                    clean_state[k] = torch.cat([ckpt_w, pad], dim=1)

        missing = set(diffusion.state_dict().keys()) - set(clean_state.keys())
        unexpected = set(clean_state.keys()) - set(diffusion.state_dict().keys())
        if missing:
            print(f"  Missing keys (initializing from model): {len(missing)}")
            for k in missing:
                clean_state[k] = diffusion.state_dict()[k]
        if unexpected:
            print(f"  Unexpected keys (ignoring): {len(unexpected)}")

        diffusion.load_state_dict(clean_state, strict=False)
    elif 'model' in checkpoint:
        diffusion.load_state_dict(checkpoint['model'], strict=True)
    else:
        raise ValueError("No model weights found")

    diffusion.eval()
    print(f"V2模型加载成功: step={checkpoint.get('step', 'unknown')}")
    return diffusion


def apply_postprocessing(scenarios_array, num_scenarios):
    """P0: enforce solar=0 at night (hours < 6 or >= 19)."""
    absolute_hours = np.tile(np.arange(SEQ_LEN), num_scenarios)
    night_mask = (absolute_hours < 6) | (absolute_hours >= 19)
    night_before = scenarios_array[night_mask, 1].copy()
    scenarios_array[night_mask, 1] = 0.0
    n_clipped = int((night_before > 0.01).sum())
    if n_clipped > 0:
        print(f"  P0夜间光伏截断: {n_clipped} 个值归零 (max_before={night_before.max():.4f})")
    return scenarios_array, n_clipped


def generate_at_level(diffusion, dataset, fe, extreme_level, per_variable, num_scenarios, template_day):
    """在指定极端程度生成场景 (V2: KarrasUnet1D + 37维条件)"""
    cond_37 = fe.build_extreme_condition_batch(
        batch_size=num_scenarios, day_idx=template_day,
        extreme_level=extreme_level, per_variable=per_variable, noise_std=0.02
    )
    cond_37 = torch.FloatTensor(cond_37).to(DEVICE)

    with torch.no_grad():
        generated = diffusion.sample(batch_size=num_scenarios, class_labels=cond_37,
                                     cond_scale=COND_SCALE)

    all_scenarios = []
    for i in range(num_scenarios):
        denormed = dataset.denormalize(generated[i].cpu()).numpy().T
        all_scenarios.append(denormed)
    scenarios_array = np.concatenate(all_scenarios, axis=0)
    scenarios_array, _ = apply_postprocessing(scenarios_array, num_scenarios)
    return scenarios_array


def generate_multi_level(diffusion, dataset, fe, template_day=TEMPLATE_DAY):
    """生成3级极端程度场景"""
    levels = {
        'normal': (0.0, None),
        'high_risk': (0.90, None),
        'extreme': (0.95, None),
    }
    results = {}
    for label, (level, pv) in levels.items():
        print(f"\n[{label.upper()}] extreme_level={level}")
        results[label] = generate_at_level(
            diffusion, dataset, fe, level, pv, NUM_SCENARIOS, template_day
        )
    return results


def save_results(scenarios_array, num_scenarios, output_path, extreme_level,
                 per_variable, fe, template_day, level_label=None):
    """保存CSV"""
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
        print(f"\n已保存至: {output_path}")
    return df_final


def print_comparison(multi_results, dataset):
    """打印多级极端程度对比"""
    orig = dataset.data
    print(f"\n{'='*70}")
    print(f"多级极端程度对比 (template_day={TEMPLATE_DAY})")
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

    print(f"\n--- P99 对比 ---")
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
    print("V2 采样 — KarrasUnet1D + 37dim EVT + CFG (ED-DDPM)")
    print("=" * 60)

    print("\n[1/3] 加载模型和数据...")
    diffusion = load_model(MODEL_PATH)
    fe = DailyFeatureExtractorV3(DATA_PATH)
    fe.print_summary()
    dataset = EnergyDataset1DV3(DATA_PATH, seq_len=SEQ_LEN, normalize=True,
                                 feature_extractor=fe)

    print(f"\n[2/3] 配置: day={TEMPLATE_DAY}, cond_scale={COND_SCALE}, dim={CONDITION_DIM}")
    os.makedirs(RESULTS_FOLDER, exist_ok=True)

    if MULTI_LEVEL_COMPARE:
        print(f"\n[3/3] 3级极端程度对比生成...")
        multi_results = generate_multi_level(diffusion, dataset, fe, TEMPLATE_DAY)

        print(f"\n保存结果...")
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
        print(f"\n已保存 {sum(len(multi_results[l])//24 for l in multi_results)} 场景至: {OUTPUT_PATH}")
        print_comparison(multi_results, dataset)
    else:
        print(f"\n[3/3] EVT引导生成 (extreme_level=0.95)...")
        scenarios = generate_at_level(
            diffusion, dataset, fe, 0.95, None, NUM_SCENARIOS, TEMPLATE_DAY
        )
        df = save_results(scenarios, NUM_SCENARIOS, OUTPUT_PATH, 0.95, None, fe, TEMPLATE_DAY)

    sample_save_path = f"{RESULTS_FOLDER}/final_samples.pt"
    if MULTI_LEVEL_COMPARE:
        torch.save({label: torch.FloatTensor(arr).reshape(-1, 5, 24)
                    for label, arr in multi_results.items()}, sample_save_path)
    print(f"\n独立样本已保存至: {sample_save_path}")
    print(f"\n{'='*60}")
    print("V2 采样完成!")
    print(f"{'='*60}")
