# -*- coding: utf-8 -*-
"""P4-06A: 600s long-context residual probe (frozen 4x7 features).

Chain: does the long context that explains LB142-v7 ALSO explain y - yhat_OOF?

Design (frozen features, no expansion):
- 4 segments x 7 market-state features (28 total): [-600,-300),[-300,-120),
  [-120,-60),[-60,0): mid_std, spread, depth, txvol, mid_trend, jump, autocorr
- residual r = y - beta*baseline (canonical OOF, beta per outer view)
- 3 frames through the frozen M01-A protocol: short only (7) / long only (21)
  / both (28)
- Outputs per frame: 4-outer deltas, corr(residual pred, residual),
  month-block sign stability, alpha, activity-stratified deltas
- Discovery PASS: >=3/4 outer positive AND short+long > short stable
- Formal entry: delta PSEUDO >= +0.0015
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
import polars as pl

from mscapital.models.m01a import run_m01a_outer, summarize_m01a
from mscapital.residual import CanonicalOOF

from p3_common import save_p3_features, load_p3_frame
from p4_01a_long_context import segment_features

MARKET = Path(r"D:\mscapital-forecasting\data\raw\train\market.feather")
CANONICAL = Path(r"D:\mscapital-kaggle\output\canonical_residual_oof\canonical_residual_oof.npz")
BASELINE_ROOT = Path(r"D:\mscapital-kaggle\output\c4_protocol_closed_final\clean-baseline-v2")
OUT = Path(r"D:\mscapital-kaggle\output\p4_06a_long_residual")
FEATURE_OUT = Path(r"D:\mscapital-kaggle\output\p4_06a_features")
OUT.mkdir(parents=True, exist_ok=True)
FEATURE_OUT.mkdir(parents=True, exist_ok=True)

SEG_NAMES = ["m600_300", "m300_120", "m120_60", "m60_0"]
FEAT_KINDS = ["mid_std", "spread", "depth", "txvol", "mid_trend", "jump", "ac"]


def main() -> None:
    canonical = CanonicalOOF(**{
        k: np.asarray(np.load(CANONICAL)[k]) for k in
        ("sample_id", "month", "target", "baseline_oof", "source_train_end")
    })
    canonical.validate()

    print("building train market segment features (4x7)...")
    X = segment_features(MARKET)  # rows indexed by sample_id
    X = X[canonical.sample_id]
    X = np.nan_to_num(X, nan=0.0).reshape(len(canonical.sample_id), -1)
    print(f"X: {X.shape}")

    names_all = [f"{s}_{k}" for s in SEG_NAMES for k in FEAT_KINDS]
    short_idx = [i for i, n in enumerate(names_all) if n.startswith("m60_0")]
    long_idx = [i for i, n in enumerate(names_all) if not n.startswith("m60_0")]

    frames = {
        "short": (short_idx, "p4-06a-short"),
        "long": (long_idx, "p4-06a-long"),
        "both": (list(range(28)), "p4-06a-both"),
    }
    results = {}
    for name, (idx, exp_id) in frames.items():
        vals = X[:, idx]
        names = tuple(names_all[i] for i in idx)
        fp = FEATURE_OUT / f"{name}_features.parquet"
        save_p3_features(fp, exp_id, names,
                         canonical.sample_id, canonical.month, canonical.target, vals)
        frame = load_p3_frame(fp, names)
        rdir = OUT / name
        for outer in ("PSEUDO", "H2", "T3", "T4"):
            diag = run_m01a_outer(canonical, frame, BASELINE_ROOT, rdir, outer)
            print(f"{name:5s} {outer}: delta={diag['delta_vs_baseline']:+.9f} "
                  f"alpha={diag['alpha']:.2f} score={diag['final_score']:.9f}")
        results[name] = summarize_m01a(rdir)

    print("\n=== P4-06A discovery gate ===")
    for name in frames:
        rows = results[name]["rows"]
        deltas = np.array([r["delta_vs_baseline"] for r in rows])
        pos = int((deltas > 0).sum())
        print(f"{name:5s}: mean_delta={deltas.mean():+.6f} pos_outers={pos}/4 "
              f"PSEUDO={deltas[0]:+.6f} gate_passed={results[name]['gate']['passed']}")

    # activity stratification on 'both' predictions
    print("\n=== activity stratification (both) ===")
    act = pl.read_ipc(MARKET, columns=["sample_id"]).group_by("sample_id").len().sort("sample_id")
    a_ids = act["sample_id"].to_numpy()
    a_cnt = act["len"].to_numpy().astype(np.float64)
    apos = np.searchsorted(a_ids, canonical.sample_id)
    activity = a_cnt[apos]
    for outer in ("PSEUDO", "H2", "T3", "T4"):
        p = np.load(OUT / "both" / "m01-a" / outer / "predictions.npz")
        pred, base, y = p["pred"], p["baseline_pred"], p["target"]
        p_sid = p["sample_id"]
        apos2 = np.searchsorted(a_ids, p_sid)
        act_o = a_cnt[apos2]
        for lo_, hi_, lbl in ((0.0, 0.5, "lo"), (0.5, 0.9, "mid"), (0.9, 1.0, "hi")):
            m = (act_o >= np.quantile(act_o, lo_)) & (act_o <= np.quantile(act_o, hi_))
            from mscapital.metrics import cosine_uncentered
            d = cosine_uncentered(pred[m], y[m]) - cosine_uncentered(base[m], y[m])
            print(f"  {outer} {lbl}-activity: delta={d:+.6f}")
    print("\nwritten to", OUT)


if __name__ == "__main__":
    main()
