# -*- coding: utf-8 -*-
"""
P10-RQ-PROD-SCFI: RealMLP RQ 全量生产训练 (152+73Z 特征) → test 预测.
train 0-70 全量 → test 预测 → 保存 npz
"""
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, r"D:\mscapital-kaggle\src")
from mscapital.models.realmlp import (
    RealMLPConfig, load_frame, CleanRealMLPPreprocessor,
    _build_torch_classes, RQKMeansEncoder, _make_optimizer, _EMA, _loss, _predict,
    _set_seed, flat_anneal,
)

DATA = r"D:\mscapital-forecasting\data\processed"
RAW = r"D:\mscapital-forecasting\data\raw"
OUT = r"D:\mscapital-kaggle\output\p10_rq_prod_scfi"

import os
os.makedirs(OUT, exist_ok=True)

t0 = time.time()
cfg = RealMLPConfig(epochs=30)
print("cfg epochs:", cfg.epochs, flush=True)

frame = load_frame(f"{DATA}/f0726_train_z_f32.parquet", f"{RAW}/train/label.feather")
print(f"train frame: {len(frame.sample_id):,} ({time.time()-t0:.0f}s)", flush=True)

pre = CleanRealMLPPreprocessor(feature_names=tuple(frame.features.columns), config=cfg)
pre.fit(frame.features, frame.target)
x, c = pre.transform(frame.features)
print(f"preprocessed: {x.shape} cats={c.shape[1]} selected={len(pre.selected_numeric)} ({time.time()-t0:.0f}s)", flush=True)

test = pd.read_parquet(f"{DATA}/f0726_test_z_f32.parquet").sort_values("sample_id").reset_index(drop=True)
x_te, c_te = pre.transform(test[list(frame.features.columns)])
print(f"test preprocessed: {x_te.shape} ({time.time()-t0:.0f}s)", flush=True)

# ==== 训练循环 (带实时进度) ====
torch, _, F, RealMLPRQ = _build_torch_classes()
_set_seed(cfg.seed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"device: {device} ({time.time()-t0:.0f}s)", flush=True)

y_train = np.asarray(frame.target, dtype=np.float64)
rq = RQKMeansEncoder(cfg.rq_encoder_layers, cfg.rq_vocab_size).fit(y_train)
codes_np = rq.encode(y_train)
cat_dims = [int(np.max(c[:, j]) + 2) if c.shape[1] else 1 for j in range(c.shape[1])]
model = RealMLPRQ(x.shape[1], cat_dims, cfg).to(device)
optimizer = _make_optimizer(model, cfg, torch)
ema = _EMA(model, cfg.ema_decay, torch)
xt = torch.from_numpy(x).to(device)
ct = torch.from_numpy(c).to(device)
yt = torch.from_numpy(y_train.astype(np.float32)).to(device)
codes = torch.from_numpy(codes_np).to(device)
n = len(y_train)
total_batches = (n + cfg.train_batch_size - 1) // cfg.train_batch_size
total_steps = max(total_batches * cfg.epochs, 1)
print(f"total_steps: {total_steps} (batch={cfg.train_batch_size}, epochs={cfg.epochs})", flush=True)

step = 0
for ep in range(1, cfg.epochs + 1):
    ep_t0 = time.time()
    permutation = torch.randperm(n, device=device)
    sums = np.zeros(4, dtype=np.float64)
    batches = 0
    model.train()
    for start in range(0, n, cfg.train_batch_size):
        progress = min(step / total_steps, 1.0)
        for group, base in zip(optimizer.param_groups, (20.0, 0.093, 1.0, 1.0, 0.1)):
            group["lr"] = flat_anneal(cfg.learning_rate * base, progress)
        indices = permutation[start:start + cfg.train_batch_size]
        target = yt[indices]
        noisy = target + torch.randn_like(target) * (cfg.label_noise_std * (1.0 - progress))
        optimizer.zero_grad(set_to_none=True)
        logits, pred = model(xt[indices], ct[indices], return_codes=True)
        loss, cos, mse, rq_loss = _loss(pred, noisy, logits, codes[indices], progress, cfg, torch, F)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.gradient_clip)
        optimizer.step()
        ema.update()
        step += 1
        sums += np.asarray([loss.item(), cos.item(), mse.item(), rq_loss.item()])
        batches += 1
    ep_dt = time.time() - ep_t0
    eta = ep_dt * (cfg.epochs - ep)
    print(f"  [ep {ep:02d}/{cfg.epochs}] loss={sums[0]/batches:.4f} cos={sums[1]/batches:.4f} "
          f"mse={sums[2]/batches:.4f} rq={sums[3]/batches:.4f} | {ep_dt:.0f}s/ep | ETA {eta/60:.0f}min", flush=True)

# EMA 推理
original = ema.apply()
pred = _predict(model, x_te, c_te, cfg, str(device), torch)
ema.restore(original)
np.savez(f"{OUT}/rq_scfi_test_pred.npz", pred=pred, test_ids=test["sample_id"].to_numpy())
print(f"saved rq_scfi_test_pred.npz ({len(pred):,})", flush=True)
print(f"mean={pred.mean():.2e} std={pred.std():.2e}", flush=True)
print("ALL DONE", flush=True)
