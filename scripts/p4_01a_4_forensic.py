# -*- coding: utf-8 -*-
"""P4-01(a)-4 Forensic Models: can market path descriptors explain |d| beyond
the existing 152 features?

Models (Ridge, no tuning):
  A: 152 existing features          -> |d|
  B: 50 market path descriptors     -> |d|
  C: 152 + market descriptors       -> |d|
5-fold OOF R2 / MAE / Top-|d| AUC. Key gate: C > A (incremental info).
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import polars as pl

from p4_01a_common import OUT, DESC_PATH, load_d, load_f0726


def ridge_oof(X: np.ndarray, y: np.ndarray, folds: int = 5, alpha: float = 1.0,
              seed: int = 0) -> tuple[np.ndarray, float, float, float]:
    """OOF ridge. Returns (oof_pred, oof_r2, oof_mae, top_auc)."""
    rng = np.random.default_rng(seed)
    n = len(y)
    perm = rng.permutation(n)
    folds_idx = np.array_split(perm, folds)
    pred = np.zeros(n)
    yc = y - y.mean()
    ystd = y.std()
    for f in folds_idx:
        tr = np.setdiff1d(np.arange(n), f)
        Xtr, Xva = X[tr], X[f]
        # standardize on train
        mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
        Xtr_s, Xva_s = (Xtr - mu) / sd, (Xva - mu) / sd
        A = Xtr_s.T @ Xtr_s + alpha * np.eye(Xtr_s.shape[1])
        w = np.linalg.solve(A, Xtr_s.T @ y[tr])
        pred[f] = Xva_s @ w
    r2 = 1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    mae = np.abs(y - pred).mean()
    # Top-|d| AUC: rank of pred vs actual in top 10%
    n_top = int(n * 0.10)
    actual_top = np.zeros(n, dtype=bool)
    actual_top[np.argsort(y)[-n_top:]] = True
    from numpy import argsort as _as
    ranks = _as(_as(pred))
    auc = (ranks[actual_top].mean() - (n_top - 1) / 2) / (n - n_top)
    return pred, r2, mae, auc


def main() -> None:
    ids, d, ad = load_d()
    fids, X152, fnames = load_f0726()
    assert np.array_equal(fids, ids), "f0726 id mismatch"
    desc = pl.read_parquet(DESC_PATH).sort("sample_id")
    assert np.array_equal(desc["sample_id"].to_numpy(), ids), "desc id mismatch"

    # NaN-safe: fill with column median (forensic only, no leakage via OOF)
    X152 = np.nan_to_num(X152, nan=0.0)

    names_m = [c for c in desc.columns if c != "sample_id"]
    Xm = desc.select(names_m).to_numpy().astype(np.float64)
    Xm = np.nan_to_num(Xm, nan=0.0, posinf=0.0, neginf=0.0)

    y = ad  # target: |d|

    print("=== P4-01(a)-4 Forensic Models (Ridge OOF 5-fold) ===")
    print(f"n={len(y):,}; A: 152 feats; B: {Xm.shape[1]} market desc; C: {X152.shape[1] + Xm.shape[1]}")

    rows = []
    for name, X in [("A_152", X152), ("B_market", Xm), ("C_152+market", np.hstack([X152, Xm]))]:
        pred, r2, mae, auc = ridge_oof(X, y)
        rows.append({"model": name, "oof_r2": r2, "oof_mae": mae, "top_auc": auc})
        print(f"  {name}: R2={r2:+.5f} MAE={mae:.6f} top10-AUC={auc:.4f}")

    # incremental gate: C vs A
    ra = rows[0]
    rc = rows[2]
    print(f"\n  incremental (C - A): dR2={rc['oof_r2'] - ra['oof_r2']:+.5f} "
          f"dAUC={rc['top_auc'] - ra['top_auc']:+.4f}")

    # stability across 5 different seeds for C vs A
    print("\n  stability (5 seeds, C vs A):")
    dr2s, daucs = [], []
    for s in range(5):
        _, r2a, _, auca = ridge_oof(X152, y, seed=s)
        _, r2c, _, aucc = ridge_oof(np.hstack([X152, Xm]), y, seed=s)
        dr2s.append(r2c - r2a)
        daucs.append(aucc - auca)
    dr2s, daucs = np.array(dr2s), np.array(daucs)
    print(f"    dR2: mean={dr2s.mean():+.5f} min={dr2s.min():+.5f} max={dr2s.max():+.5f}")
    print(f"    dAUC: mean={daucs.mean():+.4f} min={daucs.min():+.4f} max={daucs.max():+.4f}")

    # market-only vs 152-only comparison for interpretability
    print(f"\n  market-only vs 152-only: R2_m={rows[1]['oof_r2']:+.5f} vs R2_152={rows[0]['oof_r2']:+.5f} "
          f"(ratio {rows[1]['oof_r2'] / (rows[0]['oof_r2'] + 1e-12):.2f})")

    import json
    (OUT / "forensic_models.json").write_text(json.dumps({
        "rows": rows,
        "incremental": {"dR2": rc["oof_r2"] - ra["oof_r2"], "dAUC": rc["top_auc"] - ra["top_auc"]},
        "stability": {"dR2_mean": float(dr2s.mean()), "dR2_min": float(dr2s.min()), "dR2_max": float(dr2s.max()),
                      "dAUC_mean": float(daucs.mean()), "dAUC_min": float(daucs.min()), "dAUC_max": float(daucs.max())},
    }, indent=2))
    print(f"\nsaved -> {OUT / 'forensic_models.json'}")


if __name__ == "__main__":
    main()
