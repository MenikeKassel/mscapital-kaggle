# -*- coding: utf-8 -*-
"""C 系列 RealMLP recipe 变体训练器 (论文方向 v2 顺序).

用法: python scripts/c_recipe.py <variant>
  baseline    C-05: robust+clip + β2=0.999 (复现锚)
  no_clip     C-06: StandardScaler (无 robust+clip 对照, 量化预处理组件)
  beta2_095   C-07: robust+clip + β2=0.95 (论文回归 +22.8% 量级)

协议: PSEUDO fold (train m0-32 / eval m33-70), 严格 temporal.
纪律: 训练用 MSE, 选 checkpoint 用 GLOBAL cosine; fold-local 预处理 fit.
"""
import json
import os
import sys
import time

import numpy as np
import polars as pl
import torch
import torch.nn as nn

RAW = r"D:\mscapital-forecasting\data\raw\train"
FEAT = r"D:\mscapital-forecasting\data\processed\f0726_train.parquet"
OUT = r"D:\mscapital-kaggle\output\c_recipe"
os.makedirs(OUT, exist_ok=True)

SEED = 2026
EPOCHS = 30
BATCH = 512
LR = 1e-3
HIDDEN = 256
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def set_seed(s):
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


def cosine_uncentered(p, y):
    p = p.reshape(-1).astype(np.float64)
    y = y.reshape(-1).astype(np.float64)
    return float(p @ y / (np.sqrt(p @ p) * np.sqrt(y @ y) + 1e-30))


class RobustScaleSmoothClip:
    def fit(self, X):
        self.median = np.median(X, axis=0)
        q75, q25 = np.quantile(X, 0.75, axis=0), np.quantile(X, 0.25, axis=0)
        qd = q75 - q25
        self.factors = np.zeros_like(qd)
        ok = qd != 0
        self.factors[ok] = 1.0 / qd[ok]
        zero = qd == 0
        rng = np.max(X, axis=0) - np.min(X, axis=0)
        self.factors[zero] = np.where(rng[zero] != 0, 2.0 / rng[zero], 0.0)
        return self

    def transform(self, X):
        x = self.factors[None, :] * (X - self.median[None, :])
        return x / np.sqrt(1 + (x / 3.0) ** 2)


class StandardScale:
    """对照: mean/std 标准化, 无 smooth clip (C-06)."""

    def fit(self, X):
        self.mean = np.mean(X, axis=0)
        self.std = np.std(X, axis=0) + 1e-8
        return self

    def transform(self, X):
        return (X - self.mean[None, :]) / self.std[None, :]


class MLP(nn.Module):
    def __init__(self, n_in, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_in, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def main():
    variant = sys.argv[1] if len(sys.argv) > 1 else "baseline"
    assert variant in ("baseline", "no_clip", "beta2_095"), variant
    t0 = time.time()

    df = pl.read_parquet(FEAT)
    lab = pl.read_ipc(f"{RAW}/label.feather")
    df = df.join(lab.select(["sample_id", "month"]), on="sample_id", how="left")
    feat_cols = [c for c in df.columns if c not in ("sample_id", "target", "month")]
    X = df.select(feat_cols).to_numpy().astype(np.float32)
    y = df["target"].to_numpy().astype(np.float64)
    m = df["month"].to_numpy()

    tr = m <= 32
    X_tr, y_tr, X_ev, y_ev = X[tr], y[tr], X[~tr], y[~tr]

    pp = RobustScaleSmoothClip() if variant != "no_clip" else StandardScale()
    pp.fit(X_tr)
    X_tr = np.nan_to_num(pp.transform(X_tr).astype(np.float32), nan=0.0)
    X_ev = np.nan_to_num(pp.transform(X_ev).astype(np.float32), nan=0.0)

    betas = (0.9, 0.999) if variant != "beta2_095" else (0.9, 0.95)

    set_seed(SEED)
    model = MLP(X_tr.shape[1], HIDDEN).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, betas=betas, weight_decay=0.0)
    lossf = nn.MSELoss()

    Xt = torch.from_numpy(X_tr).to(DEVICE)
    yt = torch.from_numpy(y_tr.astype(np.float32)).to(DEVICE)
    Xe = torch.from_numpy(X_ev).to(DEVICE)
    n = len(X_tr)
    hist = []
    best_cos, best_cos_ep = -9, 0
    for ep in range(EPOCHS):
        model.train()
        perm = torch.randperm(n, device=DEVICE)
        for i in range(0, n, BATCH):
            idx = perm[i:i + BATCH]
            opt.zero_grad()
            loss = lossf(model(Xt[idx]), yt[idx])
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            pv = [model(Xe[i:i + 4096]).cpu().numpy() for i in range(0, len(Xe), 4096)]
        p_ev = np.concatenate(pv)
        cos = cosine_uncentered(p_ev, y_ev)
        mse = float(np.mean((p_ev - y_ev) ** 2))
        hist.append({"epoch": ep + 1, "cosine": cos, "mse": mse})
        if cos > best_cos:
            best_cos, best_cos_ep = cos, ep + 1
            torch.save(model.state_dict(), f"{OUT}/{variant}_best.pt")
        print(f"[{variant}] ep {ep+1:02d} cos={cos:.6f}", flush=True)

    model.load_state_dict(torch.load(f"{OUT}/{variant}_best.pt"))
    model.eval()
    with torch.no_grad():
        p_final = np.concatenate([model(Xe[i:i + 4096]).cpu().numpy() for i in range(0, len(Xe), 4096)])
    final_cos = cosine_uncentered(p_final, y_ev)

    results = {
        "variant": variant, "betas": list(betas), "preprocess": "robust+clip" if variant != "no_clip" else "standard",
        "pseudo_eval": {"train_months": "0-32", "eval_months": "33-70"},
        "best_cosine_epoch": best_cos_ep, "best_cosine": best_cos,
        "final_cosine": final_cos, "runtime_s": round(time.time() - t0, 1),
    }
    with open(f"{OUT}/{variant}_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(json.dumps({k: results[k] for k in ("variant", "best_cosine_epoch", "best_cosine", "runtime_s")}, indent=2))


if __name__ == "__main__":
    main()
