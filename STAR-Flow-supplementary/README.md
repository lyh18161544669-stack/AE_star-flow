# STAR-Flow: Spatio-Temporal Aligned Rectified Flow

Supplementary code for the paper:

> *"STAR-Flow: Fast and Calibrated Extreme Scenario Generation for Integrated Energy Systems via Flow Matching"*
> (Submitted to Energy, 2026)

## Requirements

Tested with Python 3.10 + PyTorch 2.0 on CUDA 11.8 / 12.1. CPU execution is possible but slow for diffusion sampling.

```bash
pip install -r requirements.txt
```

## File structure

```
src/
├── dual_channel_denoiser.py    # DualChannelTransformer + CrossAttnGate + STAR (V2.5–V4)
├── flow_matching.py             # Continuous-time flow matching (FM variant)
├── wgan_gp_v1.py                # WGAN-GP baseline (V1)
├── gat_condition.py             # GAT condition encoder (V4)
├── cvae_baseline.py             # ConvCVAE baseline
├── dataset_energy_v3.py         # Data loading, DailyFeatureExtractorV3 (EVT features)
├── evaluation_metrics.py        # CRPS, Tail Cal, ED, SWD, PIT, VSS, etc.
├── ies_optimizer.py             # 672-dim LP optimal dispatch for IES
├── ies_reliability_metrics.py   # LOLP, EENS, L4 cost, Slack Duration
└── denoising_diffusion_pytorch/ # DDPM backend (Karras UNet, continuous-time diffusion)
    ├── __init__.py
    ├── attend.py
    ├── continuous_time_diffusion_1d.py
    └── karras_unet_1d.py

train/
├── train_energy_continuous_v1.py       # WGAN-GP (~20min)
├── train_energy_continuous_v2.py       # Karras UNet DDPM (~3h)
├── train_energy_continuous_v2.5.py     # DualChannel + Additive + STAR (~1h)
├── train_energy_continuous_v3.py       # DualChannel + CrossAttnGate + DDPM (~1h)
├── train_energy_continuous_v3_fm.py    # STAR-Flow (FM, 50-step ODE, ~1h)
├── train_energy_continuous_v3_iesgat.py
├── train_energy_continuous_v3_fm_iesgat.py
├── train_energy_continuous_v4.py       # DualChannel + GAT condition encoder (~1h)
└── train_cvae_baseline.py              # ConvCVAE (~30min)

sample/
├── sample_energy_continuous_v1.py
├── sample_energy_continuous_v2.py
├── sample_energy_continuous_v2.5.py
├── sample_energy_continuous_v3.py
├── sample_energy_continuous_v3_fm.py
├── sample_energy_continuous_v3_iesgat.py
├── sample_energy_continuous_v3_fm_iesgat.py
├── sample_energy_continuous_v4.py
└── sample_cvae_baseline.py

scenarios/              # Pre-generated 12-day scenarios for all 9 models
results/                # Paper metrics (Tables 4.1–4.6, 5.1–5.6)
```

## Quick start

### Option A: Evaluate from pre-generated scenarios (no training needed)
```bash
python _evaluate_all_models.py
python _report_full.py
```
All paper tables and metrics are reproduced from `scenarios/`.

### Option B: Train from scratch + sample + evaluate
```bash
# Train STAR-Flow (the proposed model)
python train/train_energy_continuous_v3_fm.py

# Sample all extreme levels across 12 test days
python sample/sample_energy_continuous_v3_fm.py

# Full evaluation
python _evaluate_all_models.py
python _statistical_tests.py
```

### Option C: Single model sampling with EVT control
```bash
python sample/sample_energy_continuous_v3_fm.py
# Output: checkpoints/generated_scenarios_v3_fm.csv
```

## EVT-guided extreme level control

The sampling scripts support three extreme levels:
- `0.0` — Normal (day-type conditional mean)
- `0.90` — High risk (P90 GPD quantile)
- `0.95` — Extreme (P95 GPD quantile)

Per-variable control is also supported:
```python
fe.build_extreme_condition(day_idx=181, extreme_level=0.0,
                           per_variable={'wind': 0.99, 'cold': 0.95})
```

## Data

`源荷数据集.csv` contains 8760 hourly records of wind, solar, electric, heat, and cold loads for a campus IES (2023, GBK encoding). See `dataset_energy_v3.py` for the stratified train/val/test split by month.

