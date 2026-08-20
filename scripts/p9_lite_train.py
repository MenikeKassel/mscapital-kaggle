# -*- coding: utf-8 -*-
"""P9-Lite 三探针训练 — C-05 E0 协议 + frozen 51-70 审判窗.

协议与 c05_recipe_e0.py 逐位一致 (fold-local robust+clip, 3x256 MLP, AdamW 1e-3,
MSE, 30ep, seed 2026, best-cosine-epoch 在 eval 33-70 选).
新增: 报告 frozen 51-70 的 cosine 与月度正占比 (审判门禁), 落盘 eval 预测.

用法:
  .venv/Scripts/python.exe scripts/p9_lite_train.py --pkg A --arm base
  .venv/Scripts/python.exe scripts/p9_lite_train.py --pkg A --arm feat
  (--arm base 时剔除该包新列, 得到同一骨架下的 152 基线; --arm feat 用 152+新列)
输出: output/p9_lite/pkg{ABC}/{base,feat}/results.json + preds.npz + best_cos.pt
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import polars as pl
import torch
import torch.nn as nn

from p9_lite_build import PKG_COLS

FEAT = r"D:\mscapital-kaggle\output\p9_lite"   # 每个 pkg 的 train_aug.parquet
RAW = r"D:\mscapital-forecasting\data\raw\train"
OUT = r"D:\mscapital-kaggle\output\p9_lite"

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
    """论文 E1 预处理 (与 C-05 E0 一致; fold-local fit)."""

    def fit(self, X):
        self.median = np.median(X, axis=0)
        q75, q25 = np.quantile(X, 0.75, axis=0), np.quantile(X, 0.25, axis=0)
        qd = q75 - q25
        self.factors = np.zeros_like(qd)
        ok = qd != 0
        self.factors[ok] = 1.0 / qd[ok]
        self.factors_saved = self.factors.copy()
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
    ap = argparse.ArgumentParser()
    ap.add_argument("--pkg", choices=["A", "B", "C"], required=True)
    ap.add_argument("--arm", choices=["base", "feat"], required=True)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--file", default=None, help="覆盖特征文件路径 (默认 pkg{train_aug}.parquet)")
    ap.add_argument("--tag", default=None, help="输出目录后缀 (区分变体, 防止覆盖 canonical)")
    args = ap.parse_args()

    outd = f"{OUT}/pkg{args.pkg}/{args.arm}" \
        + ("_s" + str(args.seed) if args.seed != 2026 else "") \
        + (("_" + args.tag) if args.tag else "")
    os.makedirs(outd, exist_ok=True)   # 必须先于训练循环存在 (best_cos.pt 落盘)
    t0 = time.time()
    path = args.file or f"{FEAT}/pkg{args.pkg}/train_aug.parquet"
    df = pl.read_parquet(path)
    lab = pl.read_ipc(f"{RAW}/label.feather")
    df = df.join(lab.select(["sample_id", "month"]), on="sample_id", how="left")
    assert "target" in df.columns

    drop = PKG_COLS[args.pkg] if args.arm == "base" else []
    feat_cols = [c for c in df.columns if c not in ("sample_id", "target", "month") and c not in drop]
    assert len(feat_cols) == 152 or args.arm == "feat", f"base 应 152 特征, got {len(feat_cols)}"
    X = df.select(feat_cols).to_numpy().astype(np.float32)
    y = df["target"].to_numpy().astype(np.float64)
    m = df["month"].to_numpy()
    sid = df["sample_id"].to_numpy()

    tr = m <= 32
    X_tr, y_tr, X_ev, y_ev, m_ev, sid_ev = X[tr], y[tr], X[~tr], y[~tr], m[~tr], sid[~tr]
    print(f"[P9-{args.pkg}/{args.arm}] n_feat={len(feat_cols)} train {X_tr.shape} / eval {X_ev.shape} "
          f"| device={DEVICE}", flush=True)

    pp = RobustScaleSmoothClip().fit(X_tr)
    X_tr = pp.transform(X_tr).astype(np.float32)
    X_ev = pp.transform(X_ev).astype(np.float32)
    X_tr = np.nan_to_num(X_tr, nan=0.0, posinf=0.0, neginf=0.0)
    X_ev = np.nan_to_num(X_ev, nan=0.0, posinf=0.0, neginf=0.0)

    set_seed(args.seed)
    model = MLP(X_tr.shape[1], HIDDEN).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, betas=(0.9, 0.999), weight_decay=0.0)
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
        hist.append({"epoch": ep + 1, "cosine": cos})
        if cos > best_cos:
            best_cos, best_cos_ep = cos, ep + 1
            torch.save(model.state_dict(), f"{OUT}/pkg{args.pkg}/{args.arm}/best_cos.pt")
        print(f"  ep {ep+1:02d} cos={cos:.6f}", flush=True)

    model.load_state_dict(torch.load(f"{OUT}/pkg{args.pkg}/{args.arm}/best_cos.pt"))
    model.eval()
    with torch.no_grad():
        p_cos = np.concatenate([model(Xe[i:i + 4096]).cpu().numpy() for i in range(0, len(Xe), 4096)])

    # ---- 审判: frozen 51-70 + 月度 ----
    fr = (m_ev >= 51) & (m_ev <= 70)
    cos_eval = cosine_uncentered(p_cos, y_ev)
    cos_fr = cosine_uncentered(p_cos[fr], y_ev[fr])
    months = np.arange(33, 71)
    mc = {}
    for mm in months:
        sel = m_ev == mm
        if sel.sum() > 0:
            mc[int(mm)] = cosine_uncentered(p_cos[sel], y_ev[sel])
    pos_fr = sum(1 for mm in range(51, 71) if mc.get(mm, 0) > 0)
    months_pos, months_neg = [], []
    for mm in range(33, 71):
        (months_pos if mc.get(mm, 0) > 0 else months_neg).append(mm)

    results = {
        "pkg": args.pkg, "arm": args.arm, "seed": args.seed, "n_feat": len(feat_cols),
        "feat_file": path,
        "train_months": "0-32", "eval_months": "33-70",
        "best_cosine_epoch": best_cos_ep, "best_cosine_eval33_70": cos_eval,
        "frozen51_70_cosine": cos_fr, "frozen51_70_pos_months": pos_fr,
        "frozen51_70_total_months": 20, "months_positive": months_pos,
        "months_negative": months_neg, "monthly_cosine": {str(k): v for k, v in mc.items()},
        "runtime_s": round(time.time() - t0, 1),
    }
    with open(f"{outd}/results.json", "w") as f:
        json.dump(results, f, indent=2)
    np.savez(f"{outd}/preds.npz", sample_id=sid_ev, month=m_ev, pred=p_cos, y=y_ev)
    print(json.dumps({k: results[k] for k in (
        "arm", "n_feat", "best_cosine_epoch", "best_cosine_eval33_70",
        "frozen51_70_cosine", "frozen51_70_pos_months", "runtime_s")}, indent=2), flush=True)


if __name__ == "__main__":
    main()
