# -*- coding: utf-8 -*-
"""P9-DG: Month-Invariant Alpha (V-REx style) — month becomes a training constraint.

L = mean_m(L_m) + lambda * var_m(L_m),  L_m = per-month MSE loss within batch.
V-REx idea (arXiv 1911.08731 GroupDRO family): penalize cross-month variance of loss.
Lambda in {0.1, 0.3, 1.0} (lambda=0 == C-05 E0 baseline, already run).

Protocol identical to C-05: m0-32 train / m33-70 eval, fold-local robust+clip,
30 epochs, best-cosine-epoch selection, global uncentered cosine on eval.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import polars as pl
import torch

from c05_recipe_e0 import RobustScaleSmoothClip, MLP, cosine_uncentered, set_seed

RAW = Path(r"D:\mscapital-forecasting\data\raw\train")
FEAT = Path(r"D:\mscapital-forecasting\data\processed\f0726_train.parquet")
OUT = Path(r"D:\mscapital-kaggle\output\p9_dg")

SEED = 2026
EPOCHS = 30
BATCH = 512
LR = 1e-3
HIDDEN = 256
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lam", type=float, required=True)
    args = ap.parse_args()
    lam = args.lam
    out = OUT / f"lam_{lam}"
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    df = pl.read_parquet(FEAT)
    lab = pl.read_ipc(RAW / "label.feather")
    df = df.join(lab.select(["sample_id", "month"]), on="sample_id", how="left")
    feat_cols = [c for c in df.columns if c not in ("sample_id", "target", "month")]
    X = df.select(feat_cols).to_numpy().astype(np.float32)
    y = df["target"].to_numpy().astype(np.float64)
    m = df["month"].to_numpy()

    tr = m <= 32
    X_tr, y_tr, m_tr = X[tr], y[tr], m[tr]
    X_ev, y_ev = X[~tr], y[~tr]
    print(f"lam={lam} train {X_tr.shape} / eval {X_ev.shape} device={DEVICE}")

    pp = RobustScaleSmoothClip().fit(X_tr)
    X_tr = np.nan_to_num(pp.transform(X_tr).astype(np.float32), nan=0.0)
    X_ev = np.nan_to_num(pp.transform(X_ev).astype(np.float32), nan=0.0)

    set_seed(SEED)
    model = MLP(X_tr.shape[1], HIDDEN).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, betas=(0.9, 0.999), weight_decay=0.0)
    lossf = torch.nn.MSELoss()

    Xt = torch.from_numpy(X_tr).to(DEVICE)
    yt = torch.from_numpy(y_tr.astype(np.float32)).to(DEVICE)
    mt = torch.from_numpy(m_tr).to(DEVICE)
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
            pred = model(Xt[idx])
            tgt = yt[idx]
            if lam == 0:
                loss = lossf(pred, tgt)
            else:
                # per-month group losses within batch
                gidx = mt[idx]
                groups = torch.unique(gidx)
                g_losses = []
                for g in groups:
                    gm = gidx == g
                    if gm.sum() >= 2:
                        g_losses.append(lossf(pred[gm], tgt[gm]))
                if len(g_losses) >= 2:
                    gl = torch.stack(g_losses)
                    loss = gl.mean() + lam * gl.var(unbiased=False)
                else:
                    loss = lossf(pred, tgt)
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            pv = []
            for i in range(0, len(Xe), 4096):
                pv.append(model(Xe[i:i + 4096]).cpu().numpy())
            p_ev = np.concatenate(pv)
        cos = cosine_uncentered(p_ev, y_ev)
        hist.append({"epoch": ep + 1, "cosine": cos})
        if cos > best_cos:
            best_cos, best_cos_ep = cos, ep + 1
            torch.save(model.state_dict(), out / "best_cos.pt")
        print(f"lam={lam} ep {ep+1:02d} cos={cos:.6f}", flush=True)

    model.load_state_dict(torch.load(out / "best_cos.pt"))
    model.eval()
    with torch.no_grad():
        p_final = np.concatenate([model(Xe[i:i + 4096]).cpu().numpy()
                                  for i in range(0, len(Xe), 4096)])
    res = {"lambda": lam, "pseudo_eval": {"train_months": "0-32", "eval_months": "33-70"},
           "best_cosine_epoch": best_cos_ep, "best_cosine": best_cos,
           "final_cosine": cosine_uncentered(p_final, y_ev),
           "runtime_s": round(time.time() - t0, 1)}
    (out / "results.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(f"lam={lam} DONE best_ep={best_cos_ep} cos={best_cos:.6f} ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
