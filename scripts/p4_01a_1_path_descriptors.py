# -*- coding: utf-8 -*-
"""P4-01(a)-1 Path Descriptors: per-sample market trajectory descriptors.

Builds from test/market.feather:
  whole-sample stats + early/mid/recent segment stats (relative position
  terciles) + evolution ratios. Covers volatility / trend / spread / depth /
  imbalance / microprice path families. NaN-safe (transaction_avgprice is
  26.4% null; all aggs skip nulls, ratios guarded).
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import polars as pl

from p4_01a_common import (
    MARKET_TEST, OUT, DESC_PATH, load_market_lf, SEGMENTS, SEG_NAMES,
)


def build() -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Returns (sample_id, descriptor_matrix, names)."""
    lf = load_market_lf()

    # ---- whole-sample stats (level + path) ----
    whole = lf.group_by("sample_id").agg(
        pl.len().alias("n_snap"),
        pl.col("mid_t").mean().alias("mid_mean"),
        pl.col("mid_t").std().alias("mid_std"),
        (pl.col("mid_t").max() - pl.col("mid_t").min()).alias("mid_range"),
        pl.col("mid_t").first().alias("mid_first"),
        pl.col("mid_t").last().alias("mid_last"),
        pl.col("spread_t").mean().alias("spread_mean"),
        pl.col("spread_t").std().alias("spread_std"),
        pl.col("spread_t").max().alias("spread_max"),
        pl.col("spread_t").first().alias("spread_first"),
        pl.col("spread_t").last().alias("spread_last"),
        pl.col("depth_t").mean().alias("depth_mean"),
        pl.col("depth_t").std().alias("depth_std"),
        pl.col("depth_t").first().alias("depth_first"),
        pl.col("depth_t").last().alias("depth_last"),
        pl.col("bid_depth_t").mean().alias("bid_depth_mean"),
        pl.col("ask_depth_t").mean().alias("ask_depth_mean"),
        pl.col("imb_t").mean().alias("imb_mean"),
        pl.col("imb_t").std().alias("imb_std"),
        pl.col("imb_t").first().alias("imb_first"),
        pl.col("imb_t").last().alias("imb_last"),
    ).with_columns(
        ((pl.col("mid_last") - pl.col("mid_first")) / (pl.col("mid_first").abs() + 1e-12)).alias("mid_trend"),
        ((pl.col("spread_last") - pl.col("spread_first")) / (pl.col("spread_first") + 1e-12)).alias("spread_trend"),
        (pl.col("depth_last") / (pl.col("depth_first") + 1e-12)).alias("depth_trend"),
        (pl.col("imb_last") - pl.col("imb_first")).alias("imb_disp"),
        (pl.col("mid_std") / (pl.col("mid_first").abs() + 1e-12)).alias("mid_vol_rel"),
        # spread widening frequency: share of snapshots where spread > first spread
    )
    # spread widening frequency needs row-level comparison -> separate pass
    widen = (
        lf.group_by("sample_id")
        .agg(pl.col("spread_t").first().alias("sp_first"))
        .join(lf.group_by("sample_id").agg(
            ((pl.col("spread_t") > pl.col("spread_t").first()).cast(pl.Int8)).mean().alias("spread_widen_freq"),
            # depth depletion: share of snapshots with depth < first depth
            ((pl.col("depth_t") < pl.col("depth_t").first()).cast(pl.Int8)).mean().alias("depth_depletion_freq"),
            # microprice displacement proxy: mid vs transaction_avgprice gap (NaN-safe)
            (pl.col("mid_t") - pl.col("transaction_avgprice")).fill_null(0.0).mean().alias("mid_avgprice_gap"),
            # mid return autocorrelation (lag-1 over row order)
            (pl.col("mid_t").diff(1) * pl.col("mid_t").diff(1).shift(-1)).fill_null(0.0).mean().alias("mid_ret_acf1"),
        ), on="sample_id")
        .select(["sample_id", "spread_widen_freq", "depth_depletion_freq", "mid_avgprice_gap", "mid_ret_acf1"])
    )
    whole = whole.join(widen, on="sample_id", how="left")

    # ---- segment stats (early/mid/recent by relative position) ----
    seg = (
        lf.with_columns(pl.int_range(0, pl.len()).over("sample_id").alias("_rn"))
        .with_columns((pl.col("_rn") / pl.len().over("sample_id")).alias("_frac"))
        .with_columns(
            pl.when(pl.col("_frac") < SEGMENTS[0][1]).then(pl.lit("early"))
            .when(pl.col("_frac") < SEGMENTS[1][1]).then(pl.lit("mid"))
            .otherwise(pl.lit("recent")).alias("_seg")
        )
        .group_by(["sample_id", "_seg"])
        .agg(
            pl.col("mid_t").mean().alias("mid_seg_mean"),
            pl.col("mid_t").std().alias("mid_seg_std"),
            pl.col("spread_t").mean().alias("spread_seg_mean"),
            pl.col("depth_t").mean().alias("depth_seg_mean"),
            pl.col("imb_t").mean().alias("imb_seg_mean"),
        )
        .collect()
        .pivot(index="sample_id", on="_seg", values=[
            "mid_seg_mean", "mid_seg_std", "spread_seg_mean", "depth_seg_mean", "imb_seg_mean",
        ])
    )

    result = whole.collect().join(seg, on="sample_id", how="left").with_columns(
        (pl.col("mid_seg_std_recent") / (pl.col("mid_seg_std_early") + 1e-12)).alias("vol_recent_early"),
        (pl.col("spread_seg_mean_recent") / (pl.col("spread_seg_mean_early") + 1e-12)).alias("spread_recent_early"),
        (pl.col("depth_seg_mean_recent") / (pl.col("depth_seg_mean_early") + 1e-12)).alias("depth_recent_early"),
        (pl.col("imb_seg_mean_recent") - pl.col("imb_seg_mean_early")).alias("imb_recent_early"),
        (pl.col("mid_seg_mean_recent") - pl.col("mid_seg_mean_early")).alias("mid_path_drift"),
    ).sort("sample_id")

    names = [c for c in result.columns if c != "sample_id"]
    ids = result["sample_id"].to_numpy().astype(np.int64)
    vals = result.select(names).to_numpy().astype(np.float64)
    # fill any null (from join gaps) with 0
    vals = np.nan_to_num(vals, nan=0.0, posinf=0.0, neginf=0.0)
    return ids, vals, names


def main() -> None:
    print("building path descriptors (118M rows)...")
    ids, vals, names = build()
    print(f"descriptors: {vals.shape} ({len(names)} cols)")
    df = pl.DataFrame({"sample_id": ids}).with_columns(
        [pl.Series(n, vals[:, i]) for i, n in enumerate(names)]
    )
    df.write_parquet(DESC_PATH)
    print(f"saved -> {DESC_PATH}")
    # summary stats
    s = df.select([pl.col(c).null_count().alias(c) for c in names]).to_dict(as_series=False)
    print("null counts:", {k: v[0] for k, v in s.items()})


if __name__ == "__main__":
    main()
