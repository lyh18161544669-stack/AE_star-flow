"""
训练脚本 V2: 使用多维条件向量 (日级统计特征 + EVT尾部特征 + 聚类 + 季节编码)
方向A: 极端场景引导的条件扩散模型

相比原始版本的变化:
  - 条件从 start_hour (24类one-hot) 改为 37维连续条件向量
  - 包含: 日统计(19) + 聚类(5) + 季节编码(2) + EVT尾部特征(10) + 极端标识(1)
  - 新增: EVT加权损失 (非对称MSE, 尾部区域加权)
  - 模型使用 condition_dim=37 替代 num_classes=24
"""

import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from multiprocessing import cpu_count

from accelerate import Accelerator
from ema_pytorch import EMA
from tqdm.auto import tqdm

from denoising_diffusion_pytorch.karras_unet_1d import KarrasUnet1D
from denoising_diffusion_pytorch.continuous_time_diffusion_1d import ContinuousTimeGaussianDiffusion1D
from dataset_energy_v2 import EnergyDataset1DV2, DailyFeatureExtractor


def cycle(dataloader):
    while True:
        for batch in dataloader:
            yield batch


@torch.no_grad()
def compute_val_loss(diffusion, val_dl, accelerator):
    """在验证集上计算平均loss"""
    diffusion.eval()
    total_loss = 0.0
    n_batches = 0
    for data, condition in val_dl:
        data = data.to(accelerator.device)
        condition = condition.to(accelerator.device)

        with accelerator.autocast():
            loss = diffusion(data, class_labels=condition)
        total_loss += loss.item()
        n_batches += 1
    diffusion.train()
    return total_loss / n_batches if n_batches > 0 else float('inf')


if __name__ == '__main__':
    # ================= 配置 =================
    DATA_PATH = "./源荷数据集.csv"
    RESULTS_FOLDER = "./results_energy_continuous_v2"

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    SEQ_LEN = 24
    CHANNELS = 5
    CONDITION_DIM = 37            # 日统计(19) + 聚类(5) + 季节(2) + EVT(10) + 极端(1)
    BATCH_SIZE = 128
    GRADIENT_ACCUMULATE_EVERY = 2
    TRAIN_STEPS = 30000
    LEARNING_RATE = 1e-4
    EMA_DECAY = 0.999
    EMA_UPDATE_EVERY = 10
    SAVE_AND_SAMPLE_EVERY = 2000
    NUM_SAMPLES = 4
    MAX_GRAD_NORM = 1.0

    # CFG: classifier-free guidance (P0a)
    COND_DROP_PROB = 0.1           # 10% condition dropout during training

    # EVT加权损失配置
    USE_EVT_LOSS_WEIGHT = True     # 启用EVT尾部加权损失
    EVT_LOSS_LAMBDA = 2.0          # 尾部权重放大系数

    # ================= 模型构建 =================
    model = KarrasUnet1D(
        seq_len=SEQ_LEN,
        dim=192,
        dim_max=768,
        channels=CHANNELS,
        condition_dim=CONDITION_DIM,     # 多维连续条件向量
        num_downsamples=2,
        num_blocks_per_stage=2,
        attn_res=(12, 6),
        fourier_dim=16,
        attn_dim_head=64,
        dropout=0.1,
        self_condition=False,
        use_positional_encoding=True,
        cond_drop_prob=COND_DROP_PROB,   # CFG condition dropout
        use_hour_embedding=True,         # P1a: sin/cos hour channels
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
        evt_loss_lambda=EVT_LOSS_LAMBDA
    )

    # ================= 数据和优化器 =================
    # 删除旧缓存 (V1格式缺少EVT字段)
    import os as _os
    cache_path = DATA_PATH.replace('.csv', '_daily_features.pkl')
    if _os.path.exists(cache_path):
        print(f"删除旧缓存 (V2 EVT格式更新): {cache_path}")
        _os.remove(cache_path)

    # 创建特征提取器 (会缓存到 .pkl 文件)
    feature_extractor = DailyFeatureExtractor(DATA_PATH)
    feature_extractor.print_summary()

    # 将尾部权重矩阵注册到扩散模型 (用于EVT加权损失)
    if USE_EVT_LOSS_WEIGHT:
        tail_weight = feature_extractor.get_tail_weight_tensor()
        print(f"尾部权重矩阵 (5x24): 范围 [{tail_weight.min():.3f}, {tail_weight.max():.3f}]")
        diffusion.set_tail_weight_matrix(tail_weight)

    dataset = EnergyDataset1DV2(
        data_path=DATA_PATH,
        seq_len=SEQ_LEN,
        normalize=True,
        feature_extractor=feature_extractor,
        split='train'
    )
    print(f"训练集大小: {len(dataset)} 个窗口")

    dl = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        pin_memory=True,
        num_workers=min(8, cpu_count()),
        persistent_workers=True
    )

    # 验证集
    val_dataset = EnergyDataset1DV2(
        data_path=DATA_PATH,
        seq_len=SEQ_LEN,
        normalize=True,
        feature_extractor=feature_extractor,
        split='val'
    )
    print(f"验证集大小: {len(val_dataset)} 个窗口")

    val_dl = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE * 2,
        pin_memory=True,
        num_workers=min(4, cpu_count()),
        persistent_workers=True
    )

    opt = Adam(diffusion.parameters(), lr=LEARNING_RATE, betas=(0.9, 0.99))

    # ================= Accelerator =================
    accelerator = Accelerator(
        split_batches=True,
        mixed_precision='fp16'
    )

    diffusion, opt, dl, val_dl = accelerator.prepare(diffusion, opt, dl, val_dl)

    # ================= EMA =================
    ema = EMA(diffusion, beta=EMA_DECAY, update_every=EMA_UPDATE_EVERY)
    ema.to(accelerator.device)

    # ================= 训练循环 =================
    step = 0
    dl_cycle = cycle(dl)

    best_val_loss = float('inf')
    best_step = 0

    # 固定条件向量用于评估采样 (day 181 = Jul 1 测试日)
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
                data, condition = next(dl_cycle)
                data = data.to(accelerator.device)
                condition = condition.to(accelerator.device)

                with accelerator.autocast():
                    loss = diffusion(data, class_labels=condition)
                    loss = loss / GRADIENT_ACCUMULATE_EVERY
                    total_loss += loss.item()

                accelerator.backward(loss)

            pbar.set_description(f'loss: {total_loss:.4f}')

            accelerator.wait_for_everyone()
            accelerator.clip_grad_norm_(diffusion.parameters(), MAX_GRAD_NORM)

            opt.step()
            opt.zero_grad()

            accelerator.wait_for_everyone()

            step += 1
            if accelerator.is_main_process:
                ema.update()

                if step != 0 and step % SAVE_AND_SAMPLE_EVERY == 0:
                    ema.ema_model.eval()

                    import os
                    os.makedirs(RESULTS_FOLDER, exist_ok=True)

                    # 验证集loss
                    val_loss = compute_val_loss(
                        ema.ema_model, val_dl, accelerator
                    )

                    with torch.no_grad():
                        milestone = step // SAVE_AND_SAMPLE_EVERY
                        all_samples = []
                        eval_cond = sample_condition.to(accelerator.device)
                        for i in range(NUM_SAMPLES):
                            sample = ema.ema_model.sample(
                                batch_size=1,
                                class_labels=eval_cond[i:i + 1]
                            )
                            all_samples.append(sample)

                        all_samples = torch.cat(all_samples, dim=0)

                        torch.save(all_samples.cpu(),
                                   f"{RESULTS_FOLDER}/sample-{milestone}.pt")

                        data_to_save = {
                            'step': step,
                            'model': accelerator.get_state_dict(diffusion),
                            'opt': opt.state_dict(),
                            'ema': ema.state_dict(),
                            'scaler': accelerator.scaler.state_dict() if accelerator.scaler else None,
                        }
                        torch.save(data_to_save,
                                   f"{RESULTS_FOLDER}/model-{milestone}.pt")

                        if val_loss < best_val_loss:
                            best_val_loss = val_loss
                            best_step = step
                            torch.save(data_to_save,
                                       f"{RESULTS_FOLDER}/model-best.pt")
                            print(f"\n  >>> Best model (val_loss={val_loss:.4f}) at step {step}")

                        print(f"\n[Milestone {milestone}] train_loss={total_loss:.4f}, "
                              f"val_loss={val_loss:.4f}, "
                              f"sample_mean={all_samples.mean():.4f}, sample_std={all_samples.std():.4f}")

            pbar.update(1)

    accelerator.print(f'V2 训练完成! Best val_loss={best_val_loss:.4f} at step {best_step}')
