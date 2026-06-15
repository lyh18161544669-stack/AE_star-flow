"""
Train ConvCVAE baseline (1D-CNN encoder/decoder) on IES energy data.

Upgraded from MLP-CVAE:
  - 1D CNN preserves temporal structure (5, 24) instead of flattening to 120
  - Lower β=0.05 for weaker KL regularization, stronger reconstruction
  - AdamW with weight_decay=1e-5
  - 300 epochs, CosineAnnealing to 1e-5
"""
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import os, warnings
warnings.filterwarnings('ignore')

from cvae_baseline import ConvCVAE, vae_loss
from dataset_energy_v3 import DailyFeatureExtractorV3, get_monthly_stratified_split, load_energy_data
from sklearn.preprocessing import MinMaxScaler

DATA_PATH = "./源荷数据集.csv"
RESULTS_DIR = "./results_cvae_baseline"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BATCH_SIZE = 128
EPOCHS = 300
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-5
BETA = 0.05             # Weak KL: prioritize reconstruction
LATENT_DIM = 64


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # ---- Load data ----
    raw, _ = load_energy_data(DATA_PATH)
    full = raw.astype(np.float32)
    scaler = MinMaxScaler(); scaler.fit(full)
    norm = scaler.transform(full)
    days = norm.reshape(365, 24, 5)  # (365, 24, 5)

    fe = DailyFeatureExtractorV3(DATA_PATH)
    train_days, val_days, test_days = get_monthly_stratified_split(DATA_PATH)
    train_days = [int(d) for d in train_days]
    val_days = [int(d) for d in val_days]

    # Build training data: (N, 5, 24) + (N, 37)
    X_train = np.stack([days[d].transpose(1, 0) for d in train_days])  # (317, 5, 24)
    C_train = np.stack([fe.get_condition(d) for d in train_days])       # (317, 37)

    X_val = np.stack([days[d].transpose(1, 0) for d in val_days])
    C_val = np.stack([fe.get_condition(d) for d in val_days])

    print(f"Train: {X_train.shape}, Val: {X_val.shape}")
    print(f"Device: {DEVICE}")

    train_ds = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(C_train))
    val_ds = TensorDataset(torch.FloatTensor(X_val), torch.FloatTensor(C_val))
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=BATCH_SIZE * 2, shuffle=False)

    # ---- Model ----
    model = ConvCVAE(seq_len=24, channels=5, condition_dim=37, latent_dim=LATENT_DIM).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"ConvCVAE params: {n_params:,}")

    opt = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(opt, T_max=EPOCHS, eta_min=1e-5)

    # ---- Training ----
    best_val_loss = float('inf')
    best_epoch = 0

    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_total, train_recon, train_kl = 0., 0., 0.
        n_batches = 0

        for x_batch, c_batch in train_dl:
            x_batch = x_batch.to(DEVICE)  # (B, 5, 24)
            c_batch = c_batch.to(DEVICE)  # (B, 37)

            x_hat, mu, logvar = model(x_batch, c_batch)
            loss, recon, kl = vae_loss(x_hat, x_batch, mu, logvar, beta=BETA)

            opt.zero_grad()
            loss.backward()
            opt.step()

            train_total += loss.item()
            train_recon += recon.item()
            train_kl += kl.item()
            n_batches += 1

        scheduler.step()

        # Validation
        model.eval()
        val_total = 0.
        with torch.no_grad():
            for x_batch, c_batch in val_dl:
                x_batch = x_batch.to(DEVICE)
                c_batch = c_batch.to(DEVICE)
                x_hat, mu, logvar = model(x_batch, c_batch)
                loss, _, _ = vae_loss(x_hat, x_batch, mu, logvar, beta=BETA)
                val_total += loss.item()

        train_total /= n_batches
        val_total /= len(val_dl)
        lr = scheduler.get_last_lr()[0]

        if epoch % 30 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d}: train={train_total:.4f} (recon={train_recon/n_batches:.4f} "
                  f"kl={train_kl/n_batches:.4f}) val={val_total:.4f} lr={lr:.2e}")

        if val_total < best_val_loss:
            best_val_loss = val_total
            best_epoch = epoch
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'opt_state_dict': opt.state_dict(),
                'scaler_min': scaler.data_min_,
                'scaler_max': scaler.data_max_,
            }, f"{RESULTS_DIR}/model-best.pt")

    print(f"\nTraining complete! Best val_loss={best_val_loss:.4f} at epoch {best_epoch}")

    torch.save({
        'epoch': EPOCHS,
        'model_state_dict': model.state_dict(),
        'scaler_min': scaler.data_min_,
        'scaler_max': scaler.data_max_,
    }, f"{RESULTS_DIR}/model-final.pt")
    print(f"Saved to {RESULTS_DIR}/")


if __name__ == '__main__':
    main()
