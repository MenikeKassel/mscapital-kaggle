# -*- coding: utf-8 -*-
"""
P4-08B: 生产 RealMLP + cosine 主导 loss (唯一变量: lambda_cos 0.01 -> 1.0)

train 0-70 全量 → test 预测 → 保存 npz (48_realmlp_prod_local.py 的镜像,
只改 loss 权重). 与原生产版对比: realmlp_prod_test_pred.npz (MSE 主导).
"""
import sys, time
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
OUT = r"D:\mscapital-forecasting\data\processed\p12_out"

t0 = time.time()
cfg = RealMLPConfig(epochs=30, lambda_cos=1.0)   # <-- 唯一改动: 0.01 -> 1.0
print("cfg epochs:", cfg.epochs, "lambda_cos:", cfg.lambda_cos, flush=True)

frame = load_frame(f"{DATA}/f0726_train_f32.parquet", f"{RAW}/train/label.feather")
print(f"train frame: {len(frame.sample_id):,} ({time.time()-t0:.0f}s)", flush=True)

pre = CleanRealMLPPreprocessor(feature_names=tuple(frame.features.columns), config=cfg)
pre.fit(frame.features, frame.target)
x, c = pre.transform(frame.features)
print(f"preprocessed: {x.shape} cats={c.shape[1]} ({time.time()-t0:.0f}s)", flush=True)

test = pd.read_parquet(f"{DATA}/f0726_test_f32.parquet").sort_values("sample_id").reset_index(drop=True)
x_te, c_te = pre.transform(test[list(frame.features.columns)])
print(f"test preprocessed: {x_te.shape} ({time.time()-t0:.0f}s)", flush=True)

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
    model.train()
    perm = torch.randperm(n, device=device)
    ep_loss = 0.0
    t_ep = time.time()
    for i in range(0, n, cfg.train_batch_size):
        idx = perm[i:i + cfg.train_batch_size]
        progress = step / max(total_steps, 1)
        optimizer.zero_grad()
        logits, y_pred = model(xt[idx], ct[idx])
        loss, cos, mse, rq_l = _loss(y_pred, yt[idx], logits, codes[idx], progress, cfg, torch, F)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.gradient_clip)
        optimizer.step()
        ema.update(model)
        ep_loss += float(loss.item())
        step += 1
    ema.apply_shadow()
    print(f"Epoch {ep}/{cfg.epochs} | loss={ep_loss/max(total_batches,1):.6f} | "
          f"{time.time()-t_ep:.0f}s | total {time.time()-t0:.0f}s", flush=True)

pred = _predict(model, x_te, c_te, cfg, device, torch)
np.savez(f"{OUT}/realmlp_cosine_test_pred.npz", pred=pred, test_ids=test["sample_id"].to_numpy())
print(f"saved realmlp_cosine_test_pred.npz ({len(pred):,}) ({time.time()-t0:.0f}s)", flush=True)
