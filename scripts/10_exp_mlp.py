# -*- coding: utf-8 -*-
"""
Exp F1: 轻量MLP — 90特征上的NN基线 (融合组件候选)
Control: LGBM 90特征 (CV1 = 0.130204)
Treatment: MLP [90->256->256->1] MSE训练, 同CV1协议, cos评估
"""
import time
import random
import numpy as np
import polars as pl
import torch
import torch.nn as nn

random.seed(2026)
np.random.seed(2026)
torch.manual_seed(2026)
torch.cuda.manual_seed_all(2026)

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
print(f"train {X_tr.shape} valid {X_va.shape}", flush=True)

# 标准化前: NaN->0 (LightGBM支持NaN但NN不行), 极端值clip
X_tr = np.nan_to_num(X_tr, nan=0.0, posinf=1e6, neginf=-1e6).clip(-1e6, 1e6)
X_va = np.nan_to_num(X_va, nan=0.0, posinf=1e6, neginf=-1e6).clip(-1e6, 1e6)
# 标准化 (fit on train)
mu = X_tr.mean(axis=0, keepdims=True)
sd = X_tr.std(axis=0, keepdims=True) + 1e-6
X_tr = ((X_tr - mu) / sd).clip(-10, 10)
X_va = ((X_va - mu) / sd).clip(-10, 10)
y_tr_mean = y_tr.mean()
y_tr_std = y_tr.std() + 1e-6
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
model = MLP(X_tr.shape[1]).to(device)
opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=15)
lossf = nn.MSELoss()

BATCH = 2048
EPOCHS = 15
n = len(X_tr)
Xv = torch.from_numpy(X_va).to(device)
yv = torch.from_numpy(y_va).to(device)
best_cos, best_state = -1.0, None

t0 = time.time()
for ep in range(EPOCHS):
    perm = torch.randperm(n)
    model.train()
    tot = 0.0
    for i in range(0, n, BATCH):
        idx = perm[i:i+BATCH]
        xb = torch.from_numpy(X_tr[idx.numpy()]).to(device)
        yb = torch.from_numpy(y_tr_n[idx.numpy()]).to(device)
        opt.zero_grad()
        loss = lossf(model(xb), yb)
        loss.backward()
        opt.step()
        tot += loss.item() * len(idx)
    sched.step()
    model.eval()
    with torch.no_grad():
        pv = model(Xv).cpu().numpy() * y_tr_std + y_tr_mean
    c = cos_uncenter(pv, y_va)
    if c > best_cos:
        best_cos = c
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    print(f"epoch {ep+1}/{EPOCHS}: loss={tot/n:.6f} valid_cos={c:.6f} ({time.time()-t0:.0f}s)", flush=True)

print(f"\n=== Exp F1 汇总 ===\nbest valid_cos = {best_cos:.6f}\nLGBM baseline = 0.130204\nMLP - LGBM = {best_cos-0.130204:+.6f}")
torch.save(best_state, r"D:\mscapital-kaggle\output\mlp_best.pt")
print("saved mlp_best.pt")
