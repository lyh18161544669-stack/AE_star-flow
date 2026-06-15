"""
V3-FM-IESGAT: DualChannelDenoiser + CrossAttentionFusion + IES-GAT + Flow Matching

Fusion model combining:
  - IESGraphAttention: physical-topology-constrained cross-variable message passing
  - Flow Matching: OT-VP velocity field prediction, 50-step ODE sampling
  - CrossAttentionFusion: bidirectional cross-attention + gated fusion

Key ablations:
  V3-FM (STAR+FM) vs V3-FM-IESGAT (IES-GAT+FM): physical topology effect under FM
  V3-IESGAT (IES-GAT+DDPM) vs V3-FM-IESGAT (IES-GAT+FM): diffusion framework effect under IES-GAT
"""
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from multiprocessing import cpu_count

from accelerate import Accelerator
from ema_pytorch import EMA
from tqdm.auto import tqdm

from dual_channel_denoiser import DualChannelDenoiser
from flow_matching import ContinuousTimeFlowMatching1D
from dataset_energy_v3 import EnergyDataset1DV3, DailyFeatureExtractorV3


def cycle(dataloader):
    while True:
        for batch in dataloader:
            yield batch


@torch.no_grad()
def compute_val_loss(diffusion, val_dl, accelerator):
    diffusion.eval()
    total_loss = 0.0
    n_batches = 0
    for data, condition_37, _node_feat in val_dl:
        data = data.to(accelerator.device)
        condition_37 = condition_37.to(accelerator.device)
        with accelerator.autocast():
            loss = diffusion(data, class_labels=condition_37)
        total_loss += loss.item()
        n_batches += 1
    diffusion.train()
    return total_loss / n_batches if n_batches > 0 else float('inf')


if __name__ == '__main__':
    DATA_PATH = "./源荷数据集.csv"
    RESULTS_FOLDER = "./results_energy_continuous_v3_fm_iesgat"

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    SEQ_LEN = 24
    CHANNELS = 5
    CONDITION_DIM = 37
    BATCH_SIZE = 128
    GRADIENT_ACCUMULATE_EVERY = 2
    TRAIN_STEPS = 50000
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-5
    EMA_DECAY = 0.995
    EMA_UPDATE_EVERY = 10
    SAVE_AND_SAMPLE_EVERY = 2000
    EARLY_STOP_PATIENCE = 5          # 5 milestones (10k steps) without improvement → stop
    NUM_SAMPLES = 4
    MAX_GRAD_NORM = 1.0

    USE_EVT_LOSS_WEIGHT = True
    EVT_LOSS_LAMBDA = 1.0
    USE_DYNAMIC_EVT_WEIGHT = True
    DYNAMIC_EVT_LAMBDA = 1.5

    DENOISER_HIDDEN_DIM = 256
    DENOISER_NUM_HEADS = 4
    DENOISER_ENCODER_LAYERS = 2
    DENOISER_DECODER_LAYERS = 4
    DENOISER_MLP_RATIO = 4
    DENOISER_DROPOUT = 0.15

    # ================= Model =================
    model = DualChannelDenoiser(
        seq_len=SEQ_LEN, channels=CHANNELS, condition_dim=CONDITION_DIM,
        hidden_dim=DENOISER_HIDDEN_DIM, num_heads=DENOISER_NUM_HEADS,
        num_encoder_layers=DENOISER_ENCODER_LAYERS,
        num_decoder_layers=DENOISER_DECODER_LAYERS,
        mlp_ratio=DENOISER_MLP_RATIO, dropout=DENOISER_DROPOUT,
        cross_var_module='ies_gat',       # IES物理拓扑图注意力
        cond_drop_prob=0.1,
        use_hour_embedding=True,
        fusion_type='cross_attn_gate',    # 交叉注意力+门控融合
        time_embed_type='fm',             # Flow Matching时间嵌入
    )

    diffusion = ContinuousTimeFlowMatching1D(
        model, seq_length=SEQ_LEN, channels=CHANNELS,
        use_evt_loss_weight=USE_EVT_LOSS_WEIGHT,
        evt_loss_lambda=EVT_LOSS_LAMBDA,
        use_dynamic_evt_weight=USE_DYNAMIC_EVT_WEIGHT,
        dynamic_evt_lambda=DYNAMIC_EVT_LAMBDA,
    )

    n_params = sum(p.numel() for p in model.parameters())
    print(f"V3-FM-IESGAT: {n_params:,} params")
    print(f"  cross_var: IES-GAT (physical topology) | fusion: cross_attn_gate | time: FM")
    print(f"  cond_dim={CONDITION_DIM}, Flow Matching OT-VP")

    # ================= Data =================
    import os as _os
    _os.makedirs(RESULTS_FOLDER, exist_ok=True)

    feature_extractor = DailyFeatureExtractorV3(DATA_PATH)
    feature_extractor.print_summary()

    from dataset_energy_v3 import get_monthly_stratified_split
    train_days, val_days, test_days = get_monthly_stratified_split(DATA_PATH)
    print(f"Split: train={len(train_days)}d, val={len(val_days)}d, test={len(test_days)}d")

    train_dataset = EnergyDataset1DV3(
        data_path=DATA_PATH, seq_len=SEQ_LEN, normalize=True,
        feature_extractor=feature_extractor, split='train'
    )
    val_dataset = EnergyDataset1DV3(
        data_path=DATA_PATH, seq_len=SEQ_LEN, normalize=True,
        feature_extractor=feature_extractor, split='val'
    )
    print(f"Train windows: {len(train_dataset)}, Val windows: {len(val_dataset)}")

    dl = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                    pin_memory=True, num_workers=min(8, cpu_count()),
                    persistent_workers=True)
    val_dl = DataLoader(val_dataset, batch_size=BATCH_SIZE * 2,
                        shuffle=False, pin_memory=True,
                        num_workers=min(4, cpu_count()))

    opt = AdamW(diffusion.parameters(), lr=LEARNING_RATE, betas=(0.9, 0.99),
                weight_decay=WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(opt, T_max=TRAIN_STEPS, eta_min=1e-5)

    accelerator = Accelerator(split_batches=True, mixed_precision='fp16')
    diffusion, opt, dl = accelerator.prepare(diffusion, opt, dl)
    val_dl = accelerator.prepare(val_dl)

    ema = EMA(diffusion, beta=EMA_DECAY, update_every=EMA_UPDATE_EVERY)
    ema.to(accelerator.device)

    if USE_EVT_LOSS_WEIGHT:
        tail_weight = feature_extractor.get_tail_weight_tensor()
        diffusion.set_tail_weight_matrix(tail_weight)
    if USE_DYNAMIC_EVT_WEIGHT:
        diffusion.register_position_quantiles(feature_extractor._position_quantiles)

    step = 0
    best_val_loss = float('inf')
    best_step = 0
    milestones_without_improvement = 0
    dl_cycle = cycle(dl)

    eval_day = 181
    sample_condition = torch.FloatTensor(
        feature_extractor.get_condition(eval_day)
    ).unsqueeze(0).repeat(NUM_SAMPLES, 1)

    with tqdm(initial=step, total=TRAIN_STEPS,
              disable=not accelerator.is_main_process) as pbar:

        while step < TRAIN_STEPS:
            diffusion.train()
            total_loss = 0.

            for _ in range(GRADIENT_ACCUMULATE_EVERY):
                data, condition_37, _node_feat = next(dl_cycle)
                data = data.to(accelerator.device)
                condition_37 = condition_37.to(accelerator.device)

                with accelerator.autocast():
                    loss = diffusion(data, class_labels=condition_37)
                    loss = loss / GRADIENT_ACCUMULATE_EVERY
                    total_loss += loss.item()

                accelerator.backward(loss)

            pbar.set_description(f'loss: {total_loss:.4f}')

            accelerator.wait_for_everyone()
            accelerator.clip_grad_norm_(diffusion.parameters(), MAX_GRAD_NORM)
            opt.step()
            scheduler.step()
            opt.zero_grad()
            accelerator.wait_for_everyone()

            step += 1
            if accelerator.is_main_process:
                ema.update()

                if step != 0 and step % SAVE_AND_SAMPLE_EVERY == 0:
                    ema.ema_model.eval()
                    _os.makedirs(RESULTS_FOLDER, exist_ok=True)

                    val_loss = compute_val_loss(ema.ema_model, val_dl, accelerator)
                    current_lr = scheduler.get_last_lr()[0]

                    with torch.no_grad():
                        milestone = step // SAVE_AND_SAMPLE_EVERY
                        all_samples_list = []
                        eval_cond = sample_condition.to(accelerator.device)

                        for i in range(NUM_SAMPLES):
                            s = ema.ema_model.sample(
                                batch_size=1, class_labels=eval_cond[i:i + 1],
                                num_steps=50, show_progress=False
                            )
                            all_samples_list.append(s)

                        all_samples = torch.cat(all_samples_list, dim=0)
                        torch.save(all_samples.cpu(), f"{RESULTS_FOLDER}/sample-{milestone}.pt")

                        data_to_save = {
                            'step': step,
                            'model': accelerator.get_state_dict(diffusion),
                            'opt': opt.state_dict(),
                            'scheduler': scheduler.state_dict(),
                            'ema': ema.state_dict(),
                            'scaler': accelerator.scaler.state_dict() if accelerator.scaler else None,
                        }
                        torch.save(data_to_save, f"{RESULTS_FOLDER}/model-{milestone}.pt")

                        if val_loss < best_val_loss:
                            best_val_loss = val_loss
                            best_step = step
                            milestones_without_improvement = 0
                            torch.save(data_to_save, f"{RESULTS_FOLDER}/model-best.pt")
                            print(f"\n  >>> Best model (val_loss={val_loss:.4f}) at step {step}")
                        else:
                            milestones_without_improvement += 1
                            if milestones_without_improvement >= EARLY_STOP_PATIENCE:
                                print(f"\n  >>> Early stop: {milestones_without_improvement} milestones "
                                      f"({milestones_without_improvement * SAVE_AND_SAMPLE_EVERY} steps) "
                                      f"without improvement")
                                break

                        print(f"\n[Milestone {milestone}] train_loss={total_loss:.4f}, "
                              f"val_loss={val_loss:.4f}, lr={current_lr:.2e}, "
                              f"sample_mean={all_samples.mean():.4f}, sample_std={all_samples.std():.4f}")

            pbar.update(1)

    accelerator.print(f'V3-FM-IESGAT 训练完成! Best val_loss={best_val_loss:.4f} at step {best_step}')
