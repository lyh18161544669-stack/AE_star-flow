"""
V4 EVT引导采样: 双通道Transformer扩散模型 + GAT条件编码器
架构: DualChannelDenoiser (STDR-DiT/AugDiT风格)

核心机制不变: extreme_level控制 + EVT尾部特征操纵
必备后处理 (P0): solar=0 当绝对小时 < 6 或 >= 19
"""
import torch
import pandas as pd
import numpy as np
import os

from dual_channel_denoiser import DualChannelDenoiser
from denoising_diffusion_pytorch.continuous_time_diffusion_1d import (
    ContinuousTimeGaussianDiffusion1D
)
from dataset_energy_v3 import DailyFeatureExtractorV3, EnergyDataset1DV3, FEATURE_COLS
from gat_condition import GATConditionEncoder

# ================= 配置 =================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEQ_LEN = 24
CHANNELS = 5
CONDITION_DIM_ORIG = 37
GAT_OUTPUT_DIM = 64
CONDITION_DIM = CONDITION_DIM_ORIG + GAT_OUTPUT_DIM  # 101
GAT_PER_VAR_FEAT = 5
NUM_SCENARIOS = 100
RESULTS_FOLDER = "./results_energy_continuous_v4"
MODEL_PATH = f"{RESULTS_FOLDER}/model-best.pt"
DATA_PATH = "./源荷数据集.csv"

TEMPLATE_DAY = 181   # Jul 1 test day
EXTREME_LEVEL = 0.95
PER_VARIABLE = None
MULTI_LEVEL_COMPARE = True
COND_SCALE = 2.0        # CFG guidance scale (V4 CFG retrained 2026-05-22)
OUTPUT_PATH = f"{RESULTS_FOLDER}/generated_scenarios_v4.csv"

# V4 双通道架构参数 (需与训练一致)
DENOISER_HIDDEN_DIM = 256
DENOISER_NUM_HEADS = 4
DENOISER_ENCODER_LAYERS = 2
DENOISER_DECODER_LAYERS = 4


def load_model_and_gat(model_path):
    """加载V4双通道扩散模型 + GAT编码器"""
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

    # V4 DualChannelDenoiser
    model = DualChannelDenoiser(
        seq_len=SEQ_LEN, channels=CHANNELS, condition_dim=CONDITION_DIM,
        hidden_dim=DENOISER_HIDDEN_DIM, num_heads=DENOISER_NUM_HEADS,
        num_encoder_layers=DENOISER_ENCODER_LAYERS,
        num_decoder_layers=DENOISER_DECODER_LAYERS,
        mlp_ratio=4, dropout=0.1, cross_var_module='star',
        use_hour_embedding=True,  # P1a
    )
    diffusion = ContinuousTimeGaussianDiffusion1D(
        model, seq_length=SEQ_LEN, channels=CHANNELS,
        noise_schedule='linear', num_sample_steps=500,
        clip_sample_denoised=True, min_snr_loss_weight=True, min_snr_gamma=5
    ).to(DEVICE)

    # GAT encoder
    gat_encoder = GATConditionEncoder(
        per_var_feat_dim=GAT_PER_VAR_FEAT,
        hidden_dim=64, num_heads=4, num_layers=2,
        output_dim=GAT_OUTPUT_DIM, dropout=0.1
    ).to(DEVICE)

    checkpoint = torch.load(model_path, map_location=DEVICE, weights_only=False)

    # Load diffusion weights
    if 'ema' in checkpoint:
        clean_state = {}
        for k, v in checkpoint['ema'].items():
            if k.startswith('ema_model.'):
                clean_state[k[len('ema_model.'):]] = v
        # CFG backward compat
        if not any('null_condition_emb' in k for k in clean_state):
            print("Old checkpoint (no CFG). CFG keys initialized randomly; cond_scale=1.0")
            for k in diffusion.state_dict():
                if 'null_condition_emb' in k:
                    clean_state[k] = diffusion.state_dict()[k]
        # P1a backward compat: pad hour embedding channels with zeros
        for k in list(clean_state.keys()):
            if 'temporal_proj.weight' in k and k in diffusion.state_dict():
                ckpt_w = clean_state[k]
                model_w = diffusion.state_dict()[k]
                if ckpt_w.shape[1] < model_w.shape[1]:
                    print(f'Old checkpoint (no hour_emb): padding {k} '
                          f'{ckpt_w.shape[1]}→{model_w.shape[1]} channels')
                    pad = torch.zeros(model_w.shape[0], model_w.shape[1] - ckpt_w.shape[1])
                    clean_state[k] = torch.cat([ckpt_w, pad], dim=1)
        diffusion.load_state_dict(clean_state, strict=True)
    elif 'model' in checkpoint:
        # CFG backward compat for non-EMA checkpoints
        ckpt_state = checkpoint['model']
        if not any('null_condition_emb' in k for k in ckpt_state):
            print("Old checkpoint (no CFG). CFG keys initialized randomly; cond_scale=1.0")
            for k in diffusion.state_dict():
                if 'null_condition_emb' in k:
                    ckpt_state[k] = diffusion.state_dict()[k]
        # P1a backward compat
        for k in list(ckpt_state.keys()):
            if 'temporal_proj.weight' in k and k in diffusion.state_dict():
                ckpt_w = ckpt_state[k]
                model_w = diffusion.state_dict()[k]
                if ckpt_w.shape[1] < model_w.shape[1]:
                    pad = torch.zeros(model_w.shape[0], model_w.shape[1] - ckpt_w.shape[1])
                    ckpt_state[k] = torch.cat([ckpt_w, pad], dim=1)
        diffusion.load_state_dict(ckpt_state, strict=True)
    else:
        raise ValueError("No model weights found")

    # Load GAT weights
    if 'ema_gat' in checkpoint:
        clean_gat = {}
        for k, v in checkpoint['ema_gat'].items():
            if k.startswith('ema_model.'):
                clean_gat[k[len('ema_model.'):]] = v
        gat_encoder.load_state_dict(clean_gat, strict=True)
    elif 'gat_encoder' in checkpoint:
        gat_encoder.load_state_dict(checkpoint['gat_encoder'], strict=True)

    diffusion.eval()
    gat_encoder.eval()
    print(f"V4模型加载成功: step={checkpoint.get('step', 'unknown')}")
    return diffusion, gat_encoder


def apply_postprocessing(scenarios_array, dataset, num_scenarios):
    """P0: enforce solar=0 at night (hours < 6 or >= 19)."""
    absolute_hours = np.tile(np.arange(SEQ_LEN), num_scenarios)
    night_mask = (absolute_hours < 6) | (absolute_hours >= 19)

    night_before = scenarios_array[night_mask, 1].copy()
    scenarios_array[night_mask, 1] = 0.0
    n_clipped = int((night_before > 0.01).sum())
    if n_clipped > 0:
        print(f"  P0夜间光伏截断: {n_clipped} 个值归零 (max_before={night_before.max():.4f})")

    return scenarios_array, n_clipped


def build_full_condition(fe, gat_encoder, day_idx, extreme_level, per_variable, batch_size):
    """构建完整101维条件: 原始37维 + GAT图嵌入64维"""
    cond_37 = fe.build_extreme_condition_batch(
        batch_size=batch_size, day_idx=day_idx,
        extreme_level=extreme_level, per_variable=per_variable, noise_std=0.02
    )
    cond_37 = torch.FloatTensor(cond_37)
    node_feat = fe.get_node_features(day_idx)
    node_feat = torch.FloatTensor(node_feat).unsqueeze(0).repeat(batch_size, 1, 1)
    return cond_37, node_feat


def generate_at_level(diffusion, gat_encoder, dataset, fe, extreme_level,
                       per_variable, num_scenarios, template_day):
    """在指定极端程度生成场景"""
    cond_37, node_feat = build_full_condition(
        fe, gat_encoder, template_day, extreme_level, per_variable, num_scenarios
    )
    cond_37 = cond_37.to(DEVICE)
    node_feat = node_feat.to(DEVICE)

    with torch.no_grad():
        graph_embed = gat_encoder(node_feat)
        condition = torch.cat([cond_37, graph_embed], dim=-1)
        generated = diffusion.sample(batch_size=num_scenarios, class_labels=condition,
                                     cond_scale=COND_SCALE)

    all_scenarios = []
    for i in range(num_scenarios):
        denormed = dataset.denormalize(generated[i].cpu()).numpy().T
        all_scenarios.append(denormed)
    scenarios_array = np.concatenate(all_scenarios, axis=0)
    scenarios_array, _ = apply_postprocessing(scenarios_array, dataset, num_scenarios)
    return scenarios_array


def generate_multi_level(diffusion, gat_encoder, dataset, fe, template_day=TEMPLATE_DAY):
    """生成多级极端程度场景"""
    levels = {
        'normal': (0.0, None),
        'high_risk': (0.90, None),
        'extreme': (0.95, None),
    }
    results = {}
    for label, (level, pv) in levels.items():
        print(f"\n[{label.upper()}] extreme_level={level}")
        results[label] = generate_at_level(
            diffusion, gat_encoder, dataset, fe, level, pv,
            NUM_SCENARIOS, template_day
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
    print("V4 EVT引导采样 — 双通道Transformer + GAT + 极端场景可控生成")
    print("=" * 60)

    print("\n[1/4] 加载模型和数据...")
    diffusion, gat_encoder = load_model_and_gat(MODEL_PATH)
    fe = DailyFeatureExtractorV3(DATA_PATH)
    fe.print_summary()
    dataset = EnergyDataset1DV3(DATA_PATH, seq_len=SEQ_LEN, normalize=True,
                                 feature_extractor=fe)

    print(f"\n[2/4] EVT引导配置")
    print(f"  模板日: {TEMPLATE_DAY}")
    print(f"  极端程度: {EXTREME_LEVEL}")
    print(f"  条件维度: {CONDITION_DIM} (原始{CONDITION_DIM_ORIG} + GAT{GAT_OUTPUT_DIM})")
    if PER_VARIABLE:
        print(f"  逐变量控制: {PER_VARIABLE}")

    os.makedirs(RESULTS_FOLDER, exist_ok=True)

    if MULTI_LEVEL_COMPARE:
        print(f"\n[3/4] 多级极端程度对比生成...")
        multi_results = generate_multi_level(diffusion, gat_encoder, dataset, fe, TEMPLATE_DAY)

        print(f"\n[4/4] 保存结果...")
        # Save each level with its proper label (P2b fix)
        level_info = {
            'normal': (0.0, 'normal'),
            'high_risk': (0.90, 'high_risk'),
            'extreme': (0.95, 'extreme'),
        }
        df_list_all = []
        for label, arr in multi_results.items():
            lvl, lbl = level_info.get(label, (EXTREME_LEVEL, label))
            n_days = len(arr) // 24
            df_lvl = save_results(arr, n_days, None, lvl, PER_VARIABLE, fe, TEMPLATE_DAY,
                                  level_label=lbl)
            df_list_all.append(df_lvl)

        combined = pd.concat(df_list_all, ignore_index=True)
        os.makedirs(os.path.dirname(OUTPUT_PATH) or ".", exist_ok=True)
        combined.to_csv(OUTPUT_PATH, index=False, encoding='utf-8-sig')
        print(f"\n已保存 {sum(len(multi_results[l])//24 for l in multi_results)} 场景至: {OUTPUT_PATH}")

        print_comparison(multi_results, dataset)
    else:
        print(f"\n[3/4] EVT引导生成 (extreme_level={EXTREME_LEVEL})...")
        scenarios = generate_at_level(
            diffusion, gat_encoder, dataset, fe, EXTREME_LEVEL, PER_VARIABLE,
            NUM_SCENARIOS, TEMPLATE_DAY
        )
        print(f"\n[4/4] 保存结果...")
        df = save_results(scenarios, NUM_SCENARIOS, OUTPUT_PATH,
                          EXTREME_LEVEL, PER_VARIABLE, fe, TEMPLATE_DAY)
        print(f"\n生成统计 (extreme_level={EXTREME_LEVEL}):")
        for c, name in enumerate(FEATURE_COLS):
            vals = scenarios[:, c]
            print(f"  {name}: mean={vals.mean():.1f}, std={vals.std():.1f}, "
                  f"min={vals.min():.1f}, max={vals.max():.1f}, "
                  f"P99={np.percentile(vals, 99):.1f}")

    sample_save_path = f"{RESULTS_FOLDER}/final_samples.pt"
    if MULTI_LEVEL_COMPARE:
        torch.save({label: torch.FloatTensor(arr).reshape(-1, 5, 24)
                    for label, arr in multi_results.items()}, sample_save_path)
    else:
        torch.save(torch.FloatTensor(scenarios).reshape(-1, 5, 24), sample_save_path)
    print(f"\n独立样本已保存至: {sample_save_path}")
    print(f"输出文件: {OUTPUT_PATH}")
    print(f"\n{'='*60}")
    print("V4 EVT引导采样完成!")
    print(f"{'='*60}")
