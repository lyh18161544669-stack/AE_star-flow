"""
V1 WGAN-GP Training: Conditional energy scenario generation.

Architecture: Generator(100+37) + Critic(5×24+37) with WGAN-GP loss.
Condition: 37-dim EVT vector (same as V2 baseline, no GAT).

Data: 源荷数据集.csv, monthly-stratified split (train=317d).
"""

import torch
from torch.utils.data import DataLoader
from multiprocessing import cpu_count
import numpy as np
import os
from tqdm import tqdm

from wgan_gp_v1 import WGAN_GP
from dataset_energy_v3 import (DailyFeatureExtractorV3, EnergyDataset1DV3,
                                get_monthly_stratified_split)
from denoising_diffusion_pytorch.continuous_time_diffusion_1d import normalize_to_neg_one_to_one


def cycle(dataloader):
    while True:
        for batch in dataloader:
            yield batch


if __name__ == '__main__':
    # ================= Configuration =================
    DATA_PATH = "./源荷数据集.csv"
    RESULTS_FOLDER = "./results_energy_continuous_v1"

    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    SEQ_LEN = 24
    CHANNELS = 5
    CONDITION_DIM = 37  # V1 uses 37-dim EVT (no GAT)
    BATCH_SIZE = 64
    G_ITERS = 30000        # generator iterations
    N_CRITIC = 5           # critic updates per generator update
    LAMBDA_GP = 10         # gradient penalty coefficient
    LR = 1e-4
    BETA1 = 0.5
    BETA2 = 0.9
    SAVE_EVERY = 2000
    NUM_SAMPLES = 4

    # ================= Model Construction =================
    wgan = WGAN_GP(
        latent_dim=100, condition_dim=CONDITION_DIM,
        channels=CHANNELS, seq_len=SEQ_LEN,
        lr=LR, beta1=BETA1, beta2=BETA2,
        n_critic=N_CRITIC, lambda_gp=LAMBDA_GP,
        device=DEVICE,
    )

    g_params = sum(p.numel() for p in wgan.G.parameters())
    d_params = sum(p.numel() for p in wgan.D.parameters())
    print(f"V1 WGAN-GP: G={g_params:,} params, D={d_params:,} params, "
          f"Total={g_params + d_params:,}")
    print(f"  Condition: {CONDITION_DIM}-dim EVT (no GAT)")
    print(f"  WGAN-GP: n_critic={N_CRITIC}, lambda_gp={LAMBDA_GP}")

    # ================= Data =================
    os.makedirs(RESULTS_FOLDER, exist_ok=True)
    cache_path = DATA_PATH.replace('.csv', '_daily_features_v3.pkl')
    if os.path.exists(cache_path):
        os.remove(cache_path)

    feature_extractor = DailyFeatureExtractorV3(DATA_PATH)
    feature_extractor.print_summary()

    train_days, val_days, test_days = get_monthly_stratified_split(DATA_PATH)
    print(f"Split: train={len(train_days)}d, val={len(val_days)}d, test={len(test_days)}d")

    train_dataset = EnergyDataset1DV3(
        data_path=DATA_PATH, seq_len=SEQ_LEN, normalize=True,
        feature_extractor=feature_extractor, split='train'
    )
    print(f"Train windows: {len(train_dataset)}")

    dl = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                    pin_memory=True,
                    num_workers=min(4, cpu_count()),
                    persistent_workers=True)
    dl_cycle = cycle(dl)

    # Fixed condition for progress samples (Jul 1 test day)
    sample_condition_37 = torch.FloatTensor(
        feature_extractor.get_condition(181)
    ).unsqueeze(0).repeat(NUM_SAMPLES, 1)

    print("V1 WGAN-GP training started!")

    # ================= Training Loop =================
    losses_log = []
    with tqdm(initial=0, total=G_ITERS) as pbar:
        while wgan.g_iters < G_ITERS:
            data, condition_37, _ = dl_cycle.__next__()

            # Normalize to [-1, 1] for WGAN-GP
            data_norm = normalize_to_neg_one_to_one(data)

            loss_info = wgan.train_step(data_norm, condition_37)
            losses_log.append(loss_info)

            pbar.update(1)
            step = wgan.g_iters

            if step % 50 == 0:
                pbar.set_description(
                    f"D={loss_info['d_loss']:.3f} "
                    f"G={loss_info['g_loss']:.3f} "
                    f"W={loss_info['wasserstein']:.3f}"
                )

            # Save checkpoint and samples
            if step % SAVE_EVERY == 0:
                milestone = step // SAVE_EVERY

                # Generate samples
                wgan.G.eval()
                with torch.no_grad():
                    # Generate in [-1,1] then convert to [0,1]
                    z = torch.randn(NUM_SAMPLES, 100, device=wgan.device)
                    c = sample_condition_37.to(wgan.device)
                    fake_neg = wgan.G(z, c)
                    fake_01 = (fake_neg + 1.0) * 0.5
                    all_samples = fake_01.cpu()
                wgan.G.train()

                torch.save(all_samples, f"{RESULTS_FOLDER}/sample-{milestone}.pt")

                # Save checkpoint
                ckpt_path = f"{RESULTS_FOLDER}/model-{milestone}.pt"
                ckpt_data = {
                    'step': step,
                    'G': wgan.G.state_dict(),
                    'D': wgan.D.state_dict(),
                    'g_optimizer': wgan.g_optimizer.state_dict(),
                    'd_optimizer': wgan.d_optimizer.state_dict(),
                    'g_iters': wgan.g_iters,
                    'data_min': train_dataset.scaler.data_min_.copy(),
                    'data_max': train_dataset.scaler.data_max_.copy(),
                    'data_range_': train_dataset.scaler.data_range_.copy(),
                    'scale_': train_dataset.scaler.scale_.copy() if hasattr(train_dataset.scaler, 'scale_') else 1.0 / train_dataset.scaler.data_range_,
                }
                torch.save(ckpt_data, ckpt_path)

                # Track best (lowest G loss in last 200 iters)
                if len(losses_log) >= 200:
                    recent_g = np.mean([l['g_loss'] for l in losses_log[-200:]])
                    if step == SAVE_EVERY or recent_g < best_g_loss:
                        best_g_loss = recent_g
                        torch.save(ckpt_data, f"{RESULTS_FOLDER}/model-best.pt")

                print(f"\n[M{milestone}] step={step}, "
                      f"D_loss={loss_info['d_loss']:.4f}, "
                      f"G_loss={loss_info['g_loss']:.4f}, "
                      f"W_dist={loss_info['wasserstein']:.4f}, "
                      f"sample_mean={all_samples.mean():.4f}")

    # Final save
    wgan.save(f"{RESULTS_FOLDER}/model-final.pt")
    print(f"\nV1 WGAN-GP training done! Best G_loss={best_g_loss:.4f}")
