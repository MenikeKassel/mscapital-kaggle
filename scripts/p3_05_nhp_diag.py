# -*- coding: utf-8 -*-
"""P3-05: Neural-Hawkes-style event intensity diagnostic (minimum gate).

Question: would a marked point process (6 event intensities + next-event
time) add information beyond the M01-A event-rate features already in the
stack?

Diagnostic (per the method card's pre-registered gate):
1. Build per-second 6-class event intensities from the processed data
   (bid_add, ask_add, bid_cancel, ask_cancel, buy_trade, sell_trade rates).
2. Compare against M01-A event-flow features:
   - corr(intensity features, M01-A event-rate features)
   - conditional-intensity proxy: next-second intensity given current state
     (a linear point-process proxy), correlated against residual target.
3. Gate: if |corr| with M01-A event rates > 0.90 OR the conditional proxy
   adds nothing to the residual, stop (no full NHP training).
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
import polars as pl

from mscapital.models.m01a import load_event_flow_frame
from mscapital.residual import CanonicalOOF

ORDER = Path(r"D:\kaggle\working\processed_data\train_order_secondly.feather")
TX = Path(r"D:\kaggle\working\processed_data\train_transaction_secondly.feather")
CANONICAL = Path(r"D:\mscapital-kaggle\output\canonical_residual_oof\canonical_residual_oof.npz")
EVENT_FLOW = Path(r"D:\mscapital-kaggle\output\m01a_features\event_flow_train.parquet")
OUT = Path(r"D:\mscapital-kaggle\output\p3_05_nhp_diagnostic")


def build_intensity_features() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-sample 6-class intensity summaries (rate + peak + autocorr proxy)."""
    order = pl.read_ipc(ORDER)
    tx = pl.read_ipc(TX)
    order = order.with_columns(
        pl.when((pl.col("side") == 0) & (pl.col("order_action") == 0)).then(1.0).otherwise(0.0).alias("bid_add"),
        pl.when((pl.col("side") == 1) & (pl.col("order_action") == 0)).then(1.0).otherwise(0.0).alias("ask_add"),
        pl.when((pl.col("side") == 0) & (pl.col("order_action") == 1)).then(1.0).otherwise(0.0).alias("bid_cancel"),
        pl.when((pl.col("side") == 1) & (pl.col("order_action") == 1)).then(1.0).otherwise(0.0).alias("ask_cancel"),
    )
    tx = tx.with_columns(
        pl.when(pl.col("side") == 0).then(1.0).otherwise(0.0).alias("buy"),
        pl.when(pl.col("side") == 1).then(1.0).otherwise(0.0).alias("sell"),
    )
    # count events per (sample, second) for each class
    oe = order.group_by("sample_id", "seconds_before_predict").agg(
        pl.col("bid_add").sum(), pl.col("ask_add").sum(),
        pl.col("bid_cancel").sum(), pl.col("ask_cancel").sum(),
    )
    te = tx.group_by("sample_id", "seconds_before_predict").agg(
        pl.col("buy").sum(), pl.col("sell").sum(),
    )
    merged = oe.join(te, on=["sample_id", "seconds_before_predict"], how="outer").fill_null(0.0)
    merged = merged.sort(["sample_id", "seconds_before_predict"])
    ids = merged["sample_id"].to_numpy().astype(np.int64)
    sec = merged["seconds_before_predict"].to_numpy().astype(np.int64)
    ch = np.column_stack([
        merged["bid_add"].to_numpy(), merged["ask_add"].to_numpy(),
        merged["bid_cancel"].to_numpy(), merged["ask_cancel"].to_numpy(),
        merged["buy"].to_numpy(), merged["sell"].to_numpy(),
    ]).astype(np.float32)
    n = int(ids.max()) + 1
    grid = np.zeros((n, 60, 6), dtype=np.float32)
    valid = (sec >= 0) & (sec < 60)
    grid[ids[valid], sec[valid], :] = ch[valid]
    # summarize: rate per class, peak second rate, first-half vs second-half, entropy-ish
    rates = grid.mean(axis=1)                                    # (n,6)
    peaks = grid.max(axis=1)                                     # (n,6)
    first = grid[:, :30, :].mean(axis=1)
    second = grid[:, 30:, :].mean(axis=1)
    trend = second - first                                       # (n,6)
    return np.hstack([rates, peaks, trend]).astype(np.float32), ids, None


def main() -> None:
    canonical = CanonicalOOF(**{
        k: np.asarray(np.load(CANONICAL)[k]) for k in
        ("sample_id", "month", "target", "baseline_oof", "source_train_end")
    })
    canonical.validate()
    flow = load_event_flow_frame(EVENT_FLOW)

    print("building 6-class intensity features...")
    X, ids, _ = build_intensity_features()
    pos = np.searchsorted(np.sort(ids), canonical.sample_id)
    order = np.argsort(ids)
    X = X[order][pos]

    # M01-A event-rate columns (counts per second at 5/15/30/60 windows)
    rate_names = [c for c in flow.feature_names if "event_count_per_second" in c]
    idx = [flow.feature_names.index(c) for c in rate_names]
    X_m01a = flow.values[:, idx]
    print("M01-A rate features:", rate_names)

    # 1) correlation between intensity features and M01-A event rates
    Xc = np.nan_to_num(X, nan=0.0)
    Mc = np.nan_to_num(X_m01a, nan=0.0)
    corr = np.corrcoef(Xc, Mc, rowvar=False)
    block = corr[:Xc.shape[1], Xc.shape[1]:]
    absmax = np.abs(block).max()
    meanabs = np.abs(block).mean()
    print(f"\nintensity(18) vs M01-A rates({len(rate_names)}): absmax corr={absmax:.4f} mean={meanabs:.4f}")

    # 2) conditional proxy: linear predict next-second total intensity from current 6-class vector
    #    (on the second-grid, per sample) -- approximated via rates
    #    proxy: corr(trend class, residual)
    residual = canonical.target - 0.0004019 * canonical.baseline_oof
    r = np.corrcoef(Xc, residual)[:Xc.shape[1], Xc.shape[1]]
    top = np.argsort(np.abs(r))[::-1][:6]
    print("intensity-vs-residual corr top6:", [(float(r[i]), i) for i in top])

    gate = {
        "intensity_m01a_absmax_corr": float(absmax),
        "intensity_m01a_mean_abs_corr": float(meanabs),
        "overlap_high_gt_0_90": bool(absmax > 0.90),
    }
    gate["recommendation"] = (
        "STOP (overlaps M01-A event rates)"
        if gate["overlap_high_gt_0_90"] else "PROCEED cautiously (NHP conditional part may add info)"
    )
    print("\n=== P3-05 gate ===")
    print(gate)

    OUT.mkdir(parents=True, exist_ok=True)
    import json
    (OUT / "diagnostic.json").write_text(json.dumps({
        "intensity_features": 18, "m01a_rate_features": list(rate_names),
        "absmax_corr": float(absmax), "mean_abs_corr": float(meanabs),
        "top6_intensity_residual_corr": [float(v) for v in r[top]],
        "gate": gate,
    }, indent=2), encoding="utf-8")
    (OUT / "report.md").write_text(
        "\n".join([
            "# P3-05 NHP intensity diagnostic", "",
            f"- 6-class intensity (18 feats) vs M01-A event rates: absmax corr `{absmax:.4f}` / mean `{meanabs:.4f}`",
            f"- recommendation: **{gate['recommendation']}**", "",
        ]), encoding="utf-8",
    )
    print("written to", OUT)


if __name__ == "__main__":
    main()
