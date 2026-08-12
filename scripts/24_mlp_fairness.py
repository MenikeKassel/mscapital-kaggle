# -*- coding: utf-8 -*-
"""
P0.5-B: MLP Fairness Check (GPT 双评审拍板)
T1/T2/T3 上 MLP 30ep vs 60ep (3seed), 判断"MLP假冠军"是否由训练不足造成
对照: 原 temporal matrix MLP 分数 T1=0.1106, T2=0.1161, T3=0.1196
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
X_all = tr.select(all_feats).to_numpy().astype(np.float32)
y_all = tr["target"].to_numpy().astype(np.float32)
m_all = tr["month"].to_numpy().astype(np.int32)

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

def fit_mlp(X, y, Xv, yv, epochs, seeds=(2026, 7, 123), batch=2048, track=False):
    Xc = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6).clip(-1e6, 1e6)
    Xvc = np.nan_to_num(Xv, nan=0.0, posinf=1e6, neginf=-1e6).clip(-1e6, 1e6)
    mu = Xc.mean(axis=0, keepdims=True); sd = Xc.std(axis=0, keepdims=True) + 1e-6
    Xs = ((Xc - mu) / sd).clip(-10, 10); Xvs = ((Xvc - mu) / sd).clip(-10, 10)
    y_mean, y_std = y.mean(), y.std() + 1e-6
    y_n = (y - y_mean) / y_std
    Xvt = torch.from_numpy(Xvs).to(device)
    yvt = torch.from_numpy(yv).to(device)
    preds, bests = [], []
    n = len(X)
    for seed in seeds:
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
        model = MLP(X.shape[1]).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
        lossf = nn.MSELoss()
        best_c, best_state = -1, None
        for ep in range(epochs):
            perm = torch.randperm(n)
            model.train()
            for i in range(0, n, batch):
                idx = perm[i:i+batch]
                xb = torch.from_numpy(Xs[idx.numpy()]).to(device)
                yb = torch.from_numpy(y_n[idx.numpy()]).to(device)
                opt.zero_grad(); loss = lossf(model(xb), yb); loss.backward(); opt.step()
            sched.step()
            if track:
                model.eval()
                with torch.no_grad():
                    pv = model(Xvt).cpu().numpy() * y_std + y_mean
                c = cos_uncenter(pv, yv)
                if c > best_c:
                    best_c = c; best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if track:
            model.load_state_dict(best_state)
            bests.append(best_c)
        model.eval()
        with torch.no_grad():
            preds.append(model(Xvt).cpu().numpy() * y_std + y_mean)
    return np.mean(preds, axis=0), bests

FOLDS = {"T1": (0, 30, 31, 40), "T2": (0, 40, 41, 50), "T3": (0, 50, 51, 60)}
T0 = time.time()
for fname, (a, b, c0, d0) in FOLDS.items():
    sel_t = (m_all >= a) & (m_all <= b); sel_v = (m_all >= c0) & (m_all <= d0)
    X, y, Xv, yv = X_all[sel_t], y_all[sel_t], X_all[sel_v], y_all[sel_v]
    print(f"\n=== {fname}: train m{a}-{b} ({X.shape[0]:,}) valid m{c0}-{d0} ===", flush=True)
    for ep in [30, 60]:
        t0 = time.time()
        pv, bests = fit_mlp(X, y, Xv, yv, ep, track=True)
        c = cos_uncenter(pv, yv)
        print(f"  {ep}ep: cos={c:.6f} (per-seed best={[f'{b:.4f}' for b in bests]}) ({time.time()-t0:.0f}s)", flush=True)

print(f"\n=== P0.5-B 汇总 (对照原矩阵: T1=0.11062, T2=0.11611, T3=0.11961) ===")
print("若 60ep 显著提升 (>+0.005) → MLP 假冠军部分由训练不足造成, 需重估")
