# -*- coding: utf-8 -*-
"""P4-HIDDEN: market.feather 600s L1/L2 snapshot features.

The 152-feature stack (0726) was built from order+tx only (60s window).
market.feather carries ~189 snapshots/sample spanning 600s of L1/L2 quotes +
per-second tx aggregates -- never used for modeling. LB142's v9 grid
(market_len=200/400 x 12ch) models exactly this stream.

This experiment: aggregate snapshot features (history segments, depth
evolution, price trend over 600s) -> frozen M01-A residual protocol.
If delta is meaningful, the 600s history layer is real and sequence
modeling of snapshots is the next step.
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

from p3_common import load_p3_frame, save_p3_features

MARKET = Path(r"D:\mscapital-forecasting\data\raw\train\market.feather")
CANONICAL = Path(r"D:\mscapital-kaggle\output\canonical_residual_oof\canonical_residual_oof.npz")
BASELINE_ROOT = Path(r"D:\mscapital-kaggle\output\c4_protocol_closed_final\clean-baseline-v2")
OUT = Path(r"D:\mscapital-kaggle\output\p4_market_history_formal")
FEATURE_OUT = Path(r"D:\mscapital-kaggle\output\p4_market_history_features")

SEGMENTS = [(60, 120), (120, 240), (240, 420), (420, 600)]


def build_market_features() -> tuple[np.ndarray, list[str]]:
    lf = pl.scan_ipc(MARKET)
    df = lf.with_columns([
        (pl.col("ask_price_1") - pl.col("bid_price_1")).alias("spread1"),
        (pl.col("ask_price_2") - pl.col("bid_price_2")).alias("spread2"),
        ((pl.col("ask_price_1") + pl.col("bid_price_1")) / 2.0).alias("mid1"),
        (pl.col("bid_volume_1") + pl.col("ask_volume_1")).alias("depth1"),
        (pl.col("bid_volume_2") + pl.col("ask_volume_2")).alias("depth2"),
    ]).collect()
    df = df.sort(["sample_id", "seconds_before_predict"])

    # first/last quote per sample (over full 600s)
    first = df.group_by("sample_id").first()
    last = df.group_by("sample_id").last()
    # per-sample aggregates
    agg = df.group_by("sample_id").agg(
        pl.col("spread1").mean().alias("spread1_mean"),
        pl.col("spread1").std().alias("spread1_std"),
        pl.col("depth1").mean().alias("depth1_mean"),
        pl.col("depth2").mean().alias("depth2_mean"),
        pl.col("transaction_volume").sum().alias("tx_vol_total"),
        pl.col("transaction_count").sum().alias("tx_cnt_total"),
        pl.col("mid1").std().alias("mid_std"),
        pl.col("mid1").mean().alias("mid_mean"),
        (pl.col("bid_volume_1") / (pl.col("bid_volume_1") + pl.col("ask_volume_1") + 1e-12)).mean().alias("imb1_mean"),
    )
    result = agg.join(first.select(["sample_id", "mid1", "depth1", "spread1"]).rename(
        {"mid1": "mid_first", "depth1": "depth1_first", "spread1": "spread1_first"}),
        on="sample_id").join(
        last.select(["sample_id", "mid1", "depth1", "spread1"]).rename(
            {"mid1": "mid_last", "depth1": "depth1_last", "spread1": "spread1_last"}),
        on="sample_id")
    result = result.with_columns([
        ((pl.col("mid_last") - pl.col("mid_first")) / (pl.col("mid_first") + 1e-12)).alias("mid_trend_600"),
        (pl.col("depth1_last") / (pl.col("depth1_first") + 1e-12)).alias("depth_trend_600"),
        ((pl.col("spread1_first") - pl.col("spread1_last")) / (pl.col("spread1_first") + 1e-12)).alias("spread_trend_600"),
    ])
    # segment stats
    seg_exprs = []
    for i, (s0, s1) in enumerate(SEGMENTS):
        seg = df.filter((pl.col("seconds_before_predict") >= s0) & (pl.col("seconds_before_predict") < s1))
        seg_agg = seg.group_by("sample_id").agg(
            pl.col("spread1").mean().alias(f"seg{i}_spread"),
            pl.col("depth1").mean().alias(f"seg{i}_depth"),
            pl.col("transaction_volume").sum().alias(f"seg{i}_vol"),
            pl.col("mid1").mean().alias(f"seg{i}_mid"),
        )
        result = result.join(seg_agg, on="sample_id", how="left")
        seg_exprs += [f"seg{i}_spread", f"seg{i}_depth", f"seg{i}_vol", f"seg{i}_mid"]
    result = result.sort("sample_id").fill_null(0.0)
    names = [c for c in result.columns if c != "sample_id"]
    values = result.select(names).to_numpy().astype(np.float32)
    return values, names


def main() -> None:
    canonical = CanonicalOOF(**{
        k: np.asarray(np.load(CANONICAL)[k]) for k in
        ("sample_id", "month", "target", "baseline_oof", "source_train_end")
    })
    canonical.validate()

    print("building market 600s snapshot features...")
    values, names = build_market_features()
    print(f"features: {values.shape} ({len(names)})")

    # align: result sorted by sample_id; canonical sample ids index directly
    ids = np.arange(values.shape[0])
    missing = set(canonical.sample_id.tolist()) - set(ids.tolist())
    if missing:
        raise ValueError(f"market features missing {len(missing)} canonical ids")
    values = values[canonical.sample_id]

    feat_path = FEATURE_OUT / "market_history_features.parquet"
    save_p3_features(
        feat_path, "p4-market-history", tuple(names),
        canonical.sample_id, canonical.month, canonical.target, values,
    )
    print("features saved")

    frame = load_p3_frame(feat_path, tuple(names))
    for outer in ("PSEUDO", "H2", "T3", "T4"):
        diag = run_m01a_outer(canonical, frame, BASELINE_ROOT, OUT, outer)
        print(f"{outer}: delta={diag['delta_vs_baseline']:+.9f} score={diag['final_score']:.9f}")

    summary = summarize_m01a(OUT)
    print("\n=== P4 market history gate ===")
    for row in summary["rows"]:
        print(f"  {row['outer']}: delta={row['delta_vs_baseline']:+.9f}")
    print(f"mean delta={summary['mean_delta']:+.9f} gate={summary['gate']}")


if __name__ == "__main__":
    main()
