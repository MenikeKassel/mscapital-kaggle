# -*- coding: utf-8 -*-
"""调试 v2: 内联复现 T1 的 MLP 训练, 观察返回值"""
import time
import random
import numpy as np
import polars as pl
import torch
import torch.nn as nn

FEAT = r"D:\mscapital-forecasting\data\processed\train_features.parquet"
torch.set_num_threads(6)

tr = pl.read_parquet(FEAT)
all_feats = [c for c in tr.columns if c not in ("sample_id", "month", "target")]
X_all = tr.select(all_feats).to_numpy().astype(np.float32)
y_all = tr["target"].to_numpy().astype(np.float32)
m_all = tr["month"].to_numpy().astype(np.int32)
sel = (m_all >= 0) & (m_all <= 30)
selv = (m_all >= 31) & (m_all <= 40)
X, y = X_all[sel], y_all[sel]
Xv, yv = X_all[selv], y_all[selv]

class MLP(nn.Module):
    def __init__(self, n_in, h=256, p=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_in, h), nn.GELU(), nn.Dropout(p),
            nn.Linear(h, h), nn.GELU(), nn.Dropout(p),
            nn.Linear(h, 1))
    def forward(self, x):
        return self.net(x).squeeze(-1)

device = torch.device("cuda")

Xc = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6).clip(-1e6, 1e6)
Xvc = np.nan_to_num(Xv, nan=0.0, posinf=1e6, neginf=-1e6).clip(-1e6, 1e6)
mu = Xc.mean(axis=0, keepdims=True); sd = Xc.std(axis=0, keepdims=True) + 1e-6
Xs = ((Xc - mu) / sd).clip(-10, 10)
Xvs = ((Xvc - mu) / sd).clip(-10, 10)
y_mean, y_std = y.mean(), y.std() + 1e-6
y_n = (y - y_mean) / y_std
Xvt = torch.from_numpy(Xvs).to(device)

preds = []
for seed in [2026, 7, 123]:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    model = MLP(X.shape[1]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=30)
    lossf = nn.MSELoss()
    n = len(X)
    for ep in range(30):
        perm = torch.randperm(n)
        model.train()
        for i in range(0, n, 2048):
            idx = perm[i:i+2048]
            xb = torch.from_numpy(Xs[idx.numpy()]).to(device)
            yb = torch.from_numpy(y_n[idx.numpy()]).to(device)
            opt.zero_grad(); loss = lossf(model(xb), yb); loss.backward(); opt.step()
        sched.step()
    model.eval()
    with torch.no_grad():
        p = model(Xvt).cpu().numpy() * y_std + y_mean
        print(f"seed {seed}: pred shape={p.shape} dtype={p.dtype} type={type(p)}", flush=True)
        preds.append(p)

r = np.mean(preds, axis=0)
print(f"return: type={type(r)} shape={getattr(r, 'shape', None)}")
