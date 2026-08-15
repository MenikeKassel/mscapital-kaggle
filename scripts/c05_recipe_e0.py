# -*- coding: utf-8 -*-
"""C-05 (E0): RealMLP recipe 冻结 baseline — plain 3×256 MLP, 纯 MSE 训练,
验证/选 checkpoint 用 GLOBAL cosine (论文纪律, 与现有 RQ 变体对照).

协议: PSEUDO fold (train m0-32 / eval m33-70), 严格 temporal.
对照: 同配置按 MSE 选 checkpoint 的诊断列 (量化选 checkpoint 指标的影响).
来源: research/paper-reading-2026-08/realmlp-lecture-and-migration.md §5 (E0).
"""
import json
import os
import time

import numpy as np
import polars as pl
import torch
import torch.nn as nn

RAW = r"D:\mscapital-forecasting\data\raw\train"
FEAT = r"D:\mscapital-forecasting\data\processed\f0726_train.parquet"
OUT = r"D:\mscapital-kaggle\output\c05_recipe_e0"
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
    """论文 E1 预处理 (E0 基线也用它, 与 C-01 一致; fold-local fit)."""

    def fit(self, X):
        self.median = np.median(X, axis=0)
        q75, q25 = np.quantile(X, 0.75, axis=0), np.quantile(X, 0.25, axis=0)
        qd = q75 - q25
        self.factors = np.zeros_like(qd)
        ok = qd != 0
        self.factors[ok] = 1.0 / qd[ok]
        # IQR=0 → min-max 退化 (论文)
        zero = qd == 0
        rng = np.max(X, axis=0) - np.min(X, axis=0)
        self.factors[zero] = np.where(rng[zero] != 0, 2.0 / rng[zero], 0.0)
        return self

    def transform(self, X):
        x = self.factors[None, :] * (X - self.median[None, :])
        return x / np.sqrt(1 + (x / 3.0) ** 2)


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
    print(f"train {X_tr.shape} / eval {X_ev.shape} | device={DEVICE}")

    # fold-local 预处理 (只 fit train)
    pp = RobustScaleSmoothClip().fit(X_tr)
    X_tr = pp.transform(X_tr).astype(np.float32)
    X_ev = pp.transform(X_ev).astype(np.float32)
    # NaN 安全: 0 填充 (f0726 已处理缺失, 防御)
    X_tr = np.nan_to_num(X_tr, nan=0.0, posinf=0.0, neginf=0.0)
    X_ev = np.nan_to_num(X_ev, nan=0.0, posinf=0.0, neginf=0.0)

    set_seed(SEED)
    model = MLP(X_tr.shape[1], HIDDEN).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, betas=(0.9, 0.999), weight_decay=0.0)
    lossf = nn.MSELoss()

    Xt = torch.from_numpy(X_tr).to(DEVICE)
    yt = torch.from_numpy(y_tr.astype(np.float32)).to(DEVICE)
    Xe = torch.from_numpy(X_ev).to(DEVICE)
    n = len(X_tr)
    hist = []
    best_cos, best_cos_ep, best_mse, best_mse_ep = -9, 0, 9e9, 0
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
            pv = []
            for i in range(0, len(Xe), 4096):
                pv.append(model(Xe[i:i + 4096]).cpu().numpy())
            p_ev = np.concatenate(pv)
        cos = cosine_uncentered(p_ev, y_ev)
        mse = float(np.mean((p_ev - y_ev) ** 2))
        hist.append({"epoch": ep + 1, "cosine": cos, "mse": mse})
        if cos > best_cos:
            best_cos, best_cos_ep = cos, ep + 1
            torch.save(model.state_dict(), f"{OUT}/best_cos.pt")
        if mse < best_mse:
            best_mse, best_mse_ep = mse, ep + 1
            torch.save(model.state_dict(), f"{OUT}/best_mse.pt")
        print(f"ep {ep+1:02d} cos={cos:.6f} mse={mse:.2e}", flush=True)

    # 最终评估: best-cosine-epoch vs best-MSE-epoch (对照)
    results = {"pseudo_eval": {"train_months": "0-32", "eval_months": "33-70"},
               "history": hist,
               "best_cosine_epoch": best_cos_ep, "best_cosine": best_cos,
               "best_mse_epoch": best_mse_ep, "best_mse": best_mse,
               "n_train": int(n), "n_eval": int(len(X_ev)),
               "runtime_s": round(time.time() - t0, 1)}
    model.load_state_dict(torch.load(f"{OUT}/best_cos.pt"))
    model.eval()
    with torch.no_grad():
        p_cos = np.concatenate([model(Xe[i:i + 4096]).cpu().numpy() for i in range(0, len(Xe), 4096)])
    results["final_cosine_bestcos"] = cosine_uncentered(p_cos, y_ev)
    model.load_state_dict(torch.load(f"{OUT}/best_mse.pt"))
    model.eval()
    with torch.no_grad():
        p_mse = np.concatenate([model(Xe[i:i + 4096]).cpu().numpy() for i in range(0, len(Xe), 4096)])
    results["final_cosine_bestmse"] = cosine_uncentered(p_mse, y_ev)
    results["delta_checkpoint_metric"] = results["final_cosine_bestcos"] - results["final_cosine_bestmse"]
    with open(f"{OUT}/results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(json.dumps({k: results[k] for k in ("best_cosine_epoch", "best_cosine", "best_mse_epoch", "best_mse",
                                               "final_cosine_bestcos", "final_cosine_bestmse",
                                               "delta_checkpoint_metric", "runtime_s")}, indent=2))


if __name__ == "__main__":
    main()
