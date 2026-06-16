# STAR-Flow: Supplementary Code and Data

Official implementation for:

> *"STAR-Flow: Flow Matching Meets Extreme Value Theory for Rapid Extreme Scenario Generation in Integrated Energy Systems"*
> (Submitted to Applied Energy, 2026)

## Requirements

Python 3.10+, PyTorch 2.0+, CUDA 11.8 / 12.1. CPU execution is possible but slow for diffusion sampling (500 SDE steps).

```bash
```

## File Structure

```
├── dual_channel_denoiser.py              # DualChannel + CrossAttnGate + STAR + IES-GAT
├── flow_matching.py                       # Continuous-time flow matching (OT-VP path, CFG)
├── wgan_gp_v1.py                          # WGAN-GP baseline (V1)
├── gat_condition.py                       # GAT condition encoder (V4, 101-dim)
├── cvae_baseline.py                       # ConvCVAE baseline (1D-CNN VAE)
├── dataset_energy_v3.py                   # Data loading, stratified split, EVT feature extraction
├── evaluation_metrics.py                  # CRPS, Tail Cal, ED, SWD, PIT, VSS, etc.
├── ies_optimizer.py                       # 672-dim IES optimal dispatch LP
├── ies_reliability_metrics.py             # LOLP, EENS, L4 cost, VSS, Slack Duration
├── _evaluate_all_models.py               # Full evaluation (9 models x 3 levels x 12 days)
├── _statistical_tests.py                  # Paired Wilcoxon + BH-FDR + Cohen's d (468 pairs)
├── _evaluate_continuous_control.py        # Continuous lambda control curves (20 levels x 6 models)
├── denoising_diffusion_pytorch/           # DDPM backend (Karras UNet, continuous-time diffusion)
│   ├── __init__.py
│   ├── attend.py
│   ├── continuous_time_diffusion_1d.py
│   └── karras_unet_1d.py
├── train/                                 # Training scripts (10 models)
│   ├── train_energy_continuous_v1.py      # WGAN-GP
│   ├── train_energy_continuous_v2.py      # Karras UNet DDPM
│   ├── train_energy_continuous_v2.5.py    # DualChannel + Additive + STAR (DDPM)
│   ├── train_energy_continuous_v3.py      # DualChannel + CrossAttnGate + STAR (DDPM)
│   ├── train_energy_continuous_v3_fm.py   # STAR-Flow (CrossAttnGate + STAR + FM, 50-step ODE)
│   ├── train_energy_continuous_v3_iesgat.py
│   ├── train_energy_continuous_v3_fm_iesgat.py
│   ├── train_energy_continuous_v4.py      # DualChannel + STAR + GAT condition (101-dim)
│   └── train_cvae_baseline.py            # ConvCVAE
├── sample/                                # Sampling scripts (9 models)
│   ├── sample_energy_continuous_v1.py
│   ├── sample_energy_continuous_v2.py
│   ├── sample_energy_continuous_v2.5.py
│   ├── sample_energy_continuous_v3.py
│   ├── sample_energy_continuous_v3_fm.py
│   ├── sample_energy_continuous_v3_iesgat.py
│   ├── sample_energy_continuous_v3_fm_iesgat.py
│   ├── sample_energy_continuous_v4.py
│   └── sample_cvae_baseline.py
└── 源荷数据集.csv                         # 8,760 hourly records (2023, GBK encoding)
```

## Quick Start

### Train from scratch + evaluate

```bash
# Train STAR-Flow (~1 h on RTX 3090)
python train/train_energy_continuous_v3_fm.py

# Sample all extreme levels across 12 test days
python sample/sample_energy_continuous_v3_fm.py

# Full evaluation
python _evaluate_all_models.py
python _statistical_tests.py
```

### Continuous control curves

```bash
python _evaluate_continuous_control.py
```

## EVT-Guided Extreme Level Control

Sampling scripts support three built-in extreme levels:
- `0.0` — Normal conditions
- `0.90` — High risk (GPD P90 tail)
- `0.95` — Extreme (GPD P95 tail)

Per-variable control is also supported:
```python
fe.build_extreme_condition(day_idx=181, extreme_level=0.0,
                           per_variable={'wind': 0.99, 'cold': 0.95})
```

## Data

`源荷数据集.csv` contains 8,760 hourly records (2023) of five energy variables — wind power, solar power, electric load, heat load, and cold load — from a campus IES in Henan Province, China (GBK encoding). The monthly stratified train/val/test split (317 / 36 / 12 days) is implemented in `dataset_energy_v3.py`.

## Reproducibility

All experiments use a fixed random seed (42) and deterministic data splits. Training hyperparameters are documented in Table 5 of the manuscript. Generated output from these scripts reproduces all tables and figures reported in the paper.

## Citation

If you use this code or data in your research, please cite:

```bibtex
@article{STAR-Flow2026,
  title   = {STAR-Flow: Flow Matching Meets Extreme Value Theory for Rapid
             Extreme Scenario Generation in Integrated Energy Systems},
  author  = {Yaohui Li and Hao Lu and Chuanxiao Zheng and Zunshi Han},
  journal = {Applied Energy},
  year    = {2026},
  note    = {Under review}
}
```

