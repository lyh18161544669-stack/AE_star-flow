"""
V4 训练脚本: 双通道Transformer扩散模型 + GAT条件编码器
架构参考: STDR-DiT / AugDiT (Xie et al., Energy 2026)

V4 相比 V3.1 的改进:
  1. DualChannelDenoiser 替换 KarrasUnet1D (80M→13.7M参数)
  2. 时序通道: Transformer Encoder + DiT Decoder over 24 time steps
  3. 特征通道: Transformer Encoder + STAR + DiT Decoder over 5 variables
  4. 保留GAT条件编码器 (37原始 + 64 GAT图嵌入 = 101维)
  5. Cosine学习率衰减 + 验证集监控
"""

import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from multiprocessing import cpu_count

from accelerate import Accelerator
from ema_pytorch import EMA
from tqdm.auto import tqdm

from dual_channel_denoiser import DualChannelDenoiser
from denoising_diffusion_pytorch.continuous_time_diffusion_1d import (
    ContinuousTimeGaussianDiffusion1D
)
from dataset_energy_v3 import EnergyDataset1DV3, DailyFeatureExtractorV3
from gat_condition import GATConditionEncoder


def cycle(dataloader):
    while True:
        for batch in dataloader:
            yield batch


@torch.no_grad()
def compute_val_loss(diffusion, gat_encoder, val_dl, accelerator):
    diffusion.eval()
    gat_encoder.eval()
    total_loss = 0.0
    n_batches = 0
    for data, condition_37, node_feat in val_dl:
        data = data.to(accelerator.device)
        condition_37 = condition_37.to(accelerator.device)
        node_feat = node_feat.to(accelerator.device)

        graph_embed = gat_encoder(node_feat)
        condition = torch.cat([condition_37, graph_embed], dim=-1)

        with accelerator.autocast():
            loss = diffusion(data, class_labels=condition)
        total_loss += loss.item()
        n_batches += 1
    diffusion.train()
    gat_encoder.train()
    return total_loss / n_batches if n_batches > 0 else float('inf')


if __name__ == '__main__':
    # ================= 配置 =================
    DATA_PATH = "./源荷数据集.csv"
    RESULTS_FOLDER = "./results_energy_continuous_v4"

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    SEQ_LEN = 24
    CHANNELS = 5
    CONDITION_DIM_ORIG = 37
    GAT_OUTPUT_DIM = 64
    CONDITION_DIM = CONDITION_DIM_ORIG + GAT_OUTPUT_DIM  # 101
    BATCH_SIZE = 128
    GRADIENT_ACCUMULATE_EVERY = 2
    TRAIN_STEPS = 30000
    LEARNING_RATE = 1e-4
    EMA_DECAY = 0.999
    EMA_UPDATE_EVERY = 10
    SAVE_AND_SAMPLE_EVERY = 2000
    NUM_SAMPLES = 4
    MAX_GRAD_NORM = 1.0

    # EVT权重
    USE_EVT_LOSS_WEIGHT = True
    EVT_LOSS_LAMBDA = 1.0
    USE_DYNAMIC_EVT_WEIGHT = True
    DYNAMIC_EVT_LAMBDA = 1.5

    # Data split: monthly-stratified (automatic)

    # GAT
    GAT_HIDDEN_DIM = 64
    GAT_NUM_HEADS = 4
    GAT_NUM_LAYERS = 2
    GAT_PER_VAR_FEAT = 5

    # V4 双通道架构
    DENOISER_HIDDEN_DIM = 256
    DENOISER_NUM_HEADS = 4
    DENOISER_ENCODER_LAYERS = 2
    DENOISER_DECODER_LAYERS = 4
    DENOISER_MLP_RATIO = 4
    DENOISER_DROPOUT = 0.1
    DENOISER_USE_STAR = True

    # ================= 模型构建 =================
    model = DualChannelDenoiser(
        seq_len=SEQ_LEN,
        channels=CHANNELS,
        condition_dim=CONDITION_DIM,
        hidden_dim=DENOISER_HIDDEN_DIM,
        num_heads=DENOISER_NUM_HEADS,
        num_encoder_layers=DENOISER_ENCODER_LAYERS,
        num_decoder_layers=DENOISER_DECODER_LAYERS,
        mlp_ratio=DENOISER_MLP_RATIO,
        dropout=DENOISER_DROPOUT,
        cross_var_module='star',     # STAR全局汇总
        cond_drop_prob=0.1,          # CFG condition dropout
        use_hour_embedding=True,     # P1a: sin/cos hour channels
    )

    gat_encoder = GATConditionEncoder(
        per_var_feat_dim=GAT_PER_VAR_FEAT,
        hidden_dim=GAT_HIDDEN_DIM,
        num_heads=GAT_NUM_HEADS,
        num_layers=GAT_NUM_LAYERS,
        output_dim=GAT_OUTPUT_DIM,
        dropout=0.1
    )

    diffusion = ContinuousTimeGaussianDiffusion1D(
        model,
        seq_length=SEQ_LEN,
        channels=CHANNELS,
        noise_schedule='linear',
        num_sample_steps=500,
        clip_sample_denoised=True,
        min_snr_loss_weight=True,
        min_snr_gamma=5,
        use_evt_loss_weight=USE_EVT_LOSS_WEIGHT,
        evt_loss_lambda=EVT_LOSS_LAMBDA,
        use_dynamic_evt_weight=USE_DYNAMIC_EVT_WEIGHT,
        dynamic_evt_lambda=DYNAMIC_EVT_LAMBDA
    )

    n_denoiser = sum(p.numel() for p in model.parameters())
    n_gat = sum(p.numel() for p in gat_encoder.parameters())
    print(f"V4 DualChannelDenoiser: {n_denoiser:,} params")
    print(f"GAT Encoder: {n_gat:,} params")
    print(f"Total: {n_denoiser + n_gat:,} params")
    print(f"条件维度: {CONDITION_DIM} (原始{CONDITION_DIM_ORIG} + GAT{GAT_OUTPUT_DIM})")

    # ================= 数据 =================
    import os as _os
    _os.makedirs(RESULTS_FOLDER, exist_ok=True)
    cache_path = DATA_PATH.replace('.csv', '_daily_features_v3.pkl')
    if _os.path.exists(cache_path):
        print(f"删除旧缓存重新计算: {cache_path}")
        _os.remove(cache_path)

    feature_extractor = DailyFeatureExtractorV3(DATA_PATH)
    feature_extractor.print_summary()

    # Monthly-stratified split: train=317d, val=36d, test=12d
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

    # ================= 优化器 =================
    opt = Adam(
        list(diffusion.parameters()) + list(gat_encoder.parameters()),
        lr=LEARNING_RATE, betas=(0.9, 0.99)
    )
    scheduler = CosineAnnealingLR(opt, T_max=TRAIN_STEPS, eta_min=1e-6)

    # ================= Accelerator =================
    accelerator = Accelerator(split_batches=True, mixed_precision='fp16')
    diffusion, gat_encoder, opt, dl = accelerator.prepare(
        diffusion, gat_encoder, opt, dl
    )
    val_dl = accelerator.prepare(val_dl)

    # ================= EMA =================
    ema = EMA(diffusion, beta=EMA_DECAY, update_every=EMA_UPDATE_EVERY)
    ema.to(accelerator.device)
    ema_gat = EMA(gat_encoder, beta=EMA_DECAY, update_every=EMA_UPDATE_EVERY)
    ema_gat.to(accelerator.device)

    # ================= EVT权重 =================
    if USE_EVT_LOSS_WEIGHT:
        tail_weight = feature_extractor.get_tail_weight_tensor()
        diffusion.set_tail_weight_matrix(tail_weight)
        print(f"V2静态尾部权重: [{tail_weight.min():.3f}, {tail_weight.max():.3f}]")

    if USE_DYNAMIC_EVT_WEIGHT:
        diffusion.register_position_quantiles(feature_extractor._position_quantiles)
        print(f"V3动态权重: lambda={DYNAMIC_EVT_LAMBDA}")

    # ================= 训练循环 =================
    step = 0
    best_val_loss = float('inf')
    best_step = 0
    dl_cycle = cycle(dl)

    eval_day = 180
    sample_condition_37 = torch.FloatTensor(
        feature_extractor.get_condition(eval_day)
    ).unsqueeze(0).repeat(NUM_SAMPLES, 1)
    sample_node_feat = torch.FloatTensor(
        feature_extractor.get_node_features(eval_day)
    ).unsqueeze(0).repeat(NUM_SAMPLES, 1, 1)

    with tqdm(initial=step, total=TRAIN_STEPS,
              disable=not accelerator.is_main_process) as pbar:

        while step < TRAIN_STEPS:
            diffusion.train()
            gat_encoder.train()

            total_loss = 0.

            for _ in range(GRADIENT_ACCUMULATE_EVERY):
                data, condition_37, node_feat = next(dl_cycle)
                data = data.to(accelerator.device)
                condition_37 = condition_37.to(accelerator.device)
                node_feat = node_feat.to(accelerator.device)

                graph_embed = gat_encoder(node_feat)
                condition = torch.cat([condition_37, graph_embed], dim=-1)

                with accelerator.autocast():
                    loss = diffusion(data, class_labels=condition)
                    loss = loss / GRADIENT_ACCUMULATE_EVERY
                    total_loss += loss.item()

                accelerator.backward(loss)

            pbar.set_description(f'loss: {total_loss:.4f}')

            accelerator.wait_for_everyone()
            accelerator.clip_grad_norm_(
                list(diffusion.parameters()) + list(gat_encoder.parameters()),
                MAX_GRAD_NORM
            )

            opt.step()
            scheduler.step()
            opt.zero_grad()

            accelerator.wait_for_everyone()

            step += 1
            if accelerator.is_main_process:
                ema.update()
                ema_gat.update()

                if step != 0 and step % SAVE_AND_SAMPLE_EVERY == 0:
                    ema.ema_model.eval()
                    ema_gat.ema_model.eval()

                    _os.makedirs(RESULTS_FOLDER, exist_ok=True)

                    val_loss = compute_val_loss(
                        ema.ema_model, ema_gat.ema_model, val_dl, accelerator
                    )
                    current_lr = scheduler.get_last_lr()[0]

                    with torch.no_grad():
                        milestone = step // SAVE_AND_SAMPLE_EVERY
                        all_samples_list = []
                        eval_cond_37 = sample_condition_37.to(accelerator.device)
                        eval_node_feat = sample_node_feat.to(accelerator.device)
                        eval_graph = ema_gat.ema_model(eval_node_feat)
                        eval_cond = torch.cat([eval_cond_37, eval_graph], dim=-1)

                        for i in range(NUM_SAMPLES):
                            s = ema.ema_model.sample(
                                batch_size=1,
                                class_labels=eval_cond[i:i + 1]
                            )
                            all_samples_list.append(s)

                        all_samples = torch.cat(all_samples_list, dim=0)
                        torch.save(all_samples.cpu(),
                                   f"{RESULTS_FOLDER}/sample-{milestone}.pt")

                        data_to_save = {
                            'step': step,
                            'model': accelerator.get_state_dict(diffusion),
                            'gat_encoder': accelerator.get_state_dict(gat_encoder),
                            'opt': opt.state_dict(),
                            'scheduler': scheduler.state_dict(),
                            'ema': ema.state_dict(),
                            'ema_gat': ema_gat.state_dict(),
                            'scaler': accelerator.scaler.state_dict() if accelerator.scaler else None,
                        }
                        torch.save(data_to_save,
                                   f"{RESULTS_FOLDER}/model-{milestone}.pt")

                        if val_loss < best_val_loss:
                            best_val_loss = val_loss
                            best_step = step
                            torch.save(data_to_save, f"{RESULTS_FOLDER}/model-best.pt")
                            print(f"\n  >>> Best model (val_loss={val_loss:.4f}) at step {step}")

                        print(f"\n[Milestone {milestone}] train_loss={total_loss:.4f}, "
                              f"val_loss={val_loss:.4f}, lr={current_lr:.2e}, "
                              f"sample_mean={all_samples.mean():.4f}, sample_std={all_samples.std():.4f}")

            pbar.update(1)

    accelerator.print(f'V4 训练完成! Best val_loss={best_val_loss:.4f} at step {best_step}')
    accelerator.print('采样时执行P0: solar=0 when hour < 6 or >= 19')
