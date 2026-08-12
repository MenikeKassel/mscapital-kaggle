# -*- coding: utf-8 -*-
"""
Exp F2: MLP 增强 — 30 epochs × 3 seeds 集成 (MLP是融合核心组件)
输出: 3个模型文件 + 集成CV
"""
import time
import random
import numpy as np
import polars as pl
import torch
import torch.nn as nn

FEAT = r"D:\mscapital-forecasting\data\processed\train_features.parquet"
N_THREADS = 12
torch.set_num_threads(N_THREADS)

tr = pl.read_parquet(FEAT)
all_feats = [c for c in tr.columns if c not in ("sample_id", "month", "target")]
tr_df = tr.filter(pl.col("month") <= 50)
va_df = tr.filter((pl.col("month") > 50) & (pl.col("month") <= 70))
X_tr = tr_df.select(all_feats).to_numpy().astype(np.float32)
y_tr = tr_df["target"].to_numpy().astype(np.float32)
X_va = va_df.select(all_feats).to_numpy().astype(np.float32)
y_va = va_df["target"].to_numpy().astype(np.float32)

X_tr_c = np.nan_to_num(X_tr, nan=0.0, posinf=1e6, neginf=-1e6).clip(-1e6, 1e6)
X_va_c = np.nan_to_num(X_va, nan=0.0, posinf=1e6, neginf=-1e6).clip(-1e6, 1e6)
mu = X_tr_c.mean(axis=0, keepdims=True)
sd = X_tr_c.std(axis=0, keepdims=True) + 1e-6
X_tr_s = ((X_tr_c - mu) / sd).clip(-10, 10)
X_va_s = ((X_va_c - mu) / sd).clip(-10, 10)
y_tr_mean = y_tr.mean(); y_tr_std = y_tr.std() + 1e-6
y_tr_n = (y_tr - y_tr_mean) / y_tr_std

def cos_uncenter(a, b):
    return float((a * b).sum() / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))

class MLP(nn.Module):
    def __init__(self, n_in, h=256, p=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_in, h), nn.GELU(), nn.Dropout(p),
            nn.Linear(h, h), nn.GELU(), nn.Dropout(p),
            nn.Linear(h, 1))
    def forward(self, x):
        return self.net(x).squeeze(-1)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"device: {device}", flush=True)
Xv = torch.from_numpy(X_va_s).to(device)
yv = torch.from_numpy(y_va).to(device)
BATCH = 2048
EPOCHS = 30
n = len(X_tr)

def train_one(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    model = MLP(X_tr.shape[1]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    lossf = nn.MSELoss()
    best_c, best_state = -1.0, None
    for ep in range(EPOCHS):
        perm = torch.randperm(n)
        model.train()
        for i in range(0, n, BATCH):
            idx = perm[i:i+BATCH]
            xb = torch.from_numpy(X_tr_s[idx.numpy()]).to(device)
            yb = torch.from_numpy(y_tr_n[idx.numpy()]).to(device)
            opt.zero_grad()
            loss = lossf(model(xb), yb)
            loss.backward()
            opt.step()
        sched.step()
        model.eval()
        with torch.no_grad():
            pv = model(Xv).cpu().numpy() * y_tr_std + y_tr_mean
        c = cos_uncenter(pv, y_va)
        if c > best_c:
            best_c = c
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    return best_c, best_state

t0 = time.time()
scores, states, preds = [], [], []
for seed in [2026, 7, 123]:
    c, st = train_one(seed)
    scores.append(c)
    states.append(st)
    print(f"seed {seed}: best_cos={c:.6f} ({time.time()-t0:.0f}s)", flush=True)
    model = MLP(X_tr.shape[1]).to(device)
    model.load_state_dict(st)
    model.eval()
    with torch.no_grad():
        preds.append(model(Xv).cpu().numpy() * y_tr_std + y_tr_mean)
    torch.save(st, rf"D:\mscapital-kaggle\output\mlp_seed{seed}.pt")

p_ens = np.mean(preds, axis=0)
c_ens = cos_uncenter(p_ens, y_va)
print(f"\n=== Exp F2 汇总 ===")
print(f"单seed: {[f'{s:.6f}' for s in scores]}")
print(f"集成(3seed): {c_ens:.6f} ({(c_ens-max(scores))*1000:+.3f}k vs best single)")
