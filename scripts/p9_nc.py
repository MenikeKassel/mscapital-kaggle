# -*- coding: utf-8 -*-
"""P9-NC: Negative Correlation Learning — train B to be accurate AND disagree with a
reference model's errors (NOT residual fitting: B still predicts y directly).

L = MSE(B, y) + lam * corr(e_B, e_p),  e_B = B(x)-y, e_p = p_ref - y (detached).
Reference p_ref = C-05 best_cos model predictions on the TRAIN fold (m0-32).
Lambda in {0.1, 0.3, 1.0} (lambda=0 == C-05 E0 baseline).

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
CKPT = Path(r"D:\mscapital-kaggle\output\c05_recipe_e0\best_cos.pt")
OUT = Path(r"D:\mscapital-kaggle\output\p9_nc")

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
    X_tr, y_tr = X[tr], y[tr]
    X_ev, y_ev = X[~tr], y[~tr]
    print(f"lam={lam} train {X_tr.shape} / eval {X_ev.shape} device={DEVICE}")

    pp = RobustScaleSmoothClip().fit(X_tr)
    X_tr = np.nan_to_num(pp.transform(X_tr).astype(np.float32), nan=0.0)
    X_ev = np.nan_to_num(pp.transform(X_ev).astype(np.float32), nan=0.0)

    # reference predictions on train fold (C-05 best_cos model)
    ref = MLP(X.shape[1])
    ref.load_state_dict(torch.load(CKPT, map_location="cpu"))
    ref.eval()
    with torch.no_grad():
        p_tr = np.concatenate([ref(torch.from_numpy(X_tr[i:i + 4096])).numpy()
                               for i in range(0, len(X_tr), 4096)]).ravel()
    print(f"reference train preds ready (n={len(p_tr)})")

    set_seed(SEED)
    model = MLP(X_tr.shape[1], HIDDEN).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, betas=(0.9, 0.999), weight_decay=0.0)
    lossf = torch.nn.MSELoss()

    Xt = torch.from_numpy(X_tr).to(DEVICE)
    yt = torch.from_numpy(y_tr.astype(np.float32)).to(DEVICE)
    pt = torch.from_numpy(p_tr.astype(np.float32)).to(DEVICE)
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
            mse = lossf(pred, tgt)
            if lam == 0:
                loss = mse
            else:
                eB = pred - tgt
                eP = pt[idx] - tgt  # detached reference error
                # NCL (Brown 2005) covariance form: penalty = mean(e_B * e_P),
                # same scale as MSE (NOT correlation [-1,1] which dominated the loss)
                loss = mse + lam * (eB * eP).mean()
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
    # diagnostics: correlation with reference errors on eval
    ref_ev = ref(torch.from_numpy(X_ev))
    with torch.no_grad():
        p_ref_ev = ref(torch.from_numpy(X_ev)).numpy().ravel()
    eB = p_final.ravel() - y_ev
    eP = p_ref_ev - y_ev
    err_corr = float(np.corrcoef(eB, eP)[0, 1])
    res = {"lambda": lam, "pseudo_eval": {"train_months": "0-32", "eval_months": "33-70"},
           "best_cosine_epoch": best_cos_ep, "best_cosine": best_cos,
           "final_cosine": cosine_uncentered(p_final, y_ev),
           "eval_error_corr_with_ref": err_corr,
           "runtime_s": round(time.time() - t0, 1)}
    (out / "results.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(f"lam={lam} DONE best_ep={best_cos_ep} cos={best_cos:.6f} "
          f"err_corr={err_corr:.4f} ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
