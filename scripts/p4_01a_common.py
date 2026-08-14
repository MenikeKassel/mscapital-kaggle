# -*- coding: utf-8 -*-
"""P4-01(a) shared: market.feather (TEST split) loaders + path descriptor builder.

|d| = unit(p_ref) - unit(p_v7) is computed on TEST (647,896 samples).
This module builds per-sample market path descriptors from test/market.feather
and aligns them with d_analysis.npz + f0726 152 features.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import polars as pl

MARKET_TEST = Path(r"D:\mscapital-forecasting\data\raw\test\market.feather")
D_ANALYSIS = Path(r"D:\mscapital-kaggle\output\p4_lb142_forensics\d_analysis.npz")
F0726_TEST = Path(r"D:\mscapital-kaggle\scripts\kaggle_0726ds\f0726_test_f32.parquet")
OUT = Path(r"D:\mscapital-kaggle\output\p4_01a")
OUT.mkdir(parents=True, exist_ok=True)

DESC_PATH = OUT / "market_path_descriptors.parquet"

# relative-position segments (fraction of each sample's snapshot count)
SEGMENTS = [(0.0, 0.333), (0.333, 0.667), (0.667, 1.0)]
SEG_NAMES = ["early", "mid", "recent"]


def load_d() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (sample_id, d, |d|) sorted by sample_id."""
    z = np.load(D_ANALYSIS)
    ids = np.asarray(z["sample_id"], dtype=np.int64)
    d = np.asarray(z["d"], dtype=np.float64)
    order = np.argsort(ids)
    return ids[order], d[order], np.abs(d[order])


def load_f0726() -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Return (sample_id, feature_matrix, feature_names) sorted by sample_id."""
    df = pl.read_parquet(F0726_TEST).sort("sample_id")
    names = [c for c in df.columns if c != "sample_id"]
    ids = df["sample_id"].to_numpy().astype(np.int64)
    X = df.select(names).to_numpy().astype(np.float64)
    return ids, X, names


def market_schema() -> pl.DataFrame:
    """Quick schema + head inspection of test market.feather."""
    lf = pl.scan_ipc(MARKET_TEST)
    cols = lf.collect_schema().names()
    df = lf.head(5).collect()
    return df.select(cols)


def load_market_lf() -> pl.LazyFrame:
    """LazyFrame of test market with derived base sequences (NaN-safe)."""
    return (
        pl.scan_ipc(MARKET_TEST)
        .with_columns(
            ((pl.col("ask_price_1") + pl.col("bid_price_1")) / 2.0).alias("mid_t"),
            (pl.col("ask_price_1") - pl.col("bid_price_1")).alias("spread_t"),
            pl.col("bid_volume_1").alias("bid_depth_t"),
            pl.col("ask_volume_1").alias("ask_depth_t"),
            (pl.col("bid_volume_1") + pl.col("ask_volume_1")).alias("depth_t"),
            (
                pl.col("bid_volume_1") / (pl.col("bid_volume_1") + pl.col("ask_volume_1") + 1e-12)
            ).alias("imb_t"),
        )
        .sort(["sample_id", "seconds_before_predict"])
    )


def _seg_stats(prefix: str, lf: pl.LazyFrame, seg_name: str) -> list[pl.Expr]:
    """Per-segment stats for a base column."""
    c = f"{prefix}_{seg_name}"
    return [
        pl.col(prefix).mean().alias(f"{c}_mean"),
        pl.col(prefix).std().alias(f"{c}_std"),
        (pl.col(prefix).max() - pl.col(prefix).min()).alias(f"{c}_range"),
    ]


def build_path_descriptors() -> tuple[np.ndarray, list[str]]:
    """Per-sample market path descriptors (whole + early/mid/recent).

    NaN handling: all aggs skip nulls; ratio features use +1e-12 guards.
    """
    lf = load_market_lf()
    n_snap = lf.group_by("sample_id").agg(pl.len().alias("n_snap"))

    # whole-sample stats
    whole = lf.group_by("sample_id").agg(
        pl.col("mid_t").mean().alias("mid_mean"),
        pl.col("mid_t").std().alias("mid_std"),
        pl.col("mid_t").max().alias("mid_max"),
        pl.col("mid_t").min().alias("mid_min"),
        pl.col("spread_t").mean().alias("spread_mean"),
        pl.col("spread_t").std().alias("spread_std"),
        pl.col("spread_t").max().alias("spread_max"),
        pl.col("depth_t").mean().alias("depth_mean"),
        pl.col("depth_t").std().alias("depth_std"),
        pl.col("imb_t").mean().alias("imb_mean"),
        pl.col("imb_t").std().alias("imb_std"),
        pl.col("bid_depth_t").mean().alias("bid_depth_mean"),
        pl.col("ask_depth_t").mean().alias("ask_depth_mean"),
        # first / last quote
        pl.col("mid_t").first().alias("mid_first"),
        pl.col("mid_t").last().alias("mid_last"),
        pl.col("spread_t").first().alias("spread_first"),
        pl.col("spread_t").last().alias("spread_last"),
        pl.col("depth_t").first().alias("depth_first"),
        pl.col("depth_t").last().alias("depth_last"),
        pl.col("imb_t").first().alias("imb_first"),
        pl.col("imb_t").last().alias("imb_last"),
    ).with_columns(
        # displacement / trend / evolution ratios (NaN-safe)
        ((pl.col("mid_last") - pl.col("mid_first")) / (pl.col("mid_first").abs() + 1e-12)).alias("mid_trend"),
        ((pl.col("spread_last") - pl.col("spread_first")) / (pl.col("spread_first") + 1e-12)).alias("spread_trend"),
        (pl.col("depth_last") / (pl.col("depth_first") + 1e-12)).alias("depth_trend"),
        (pl.col("imb_last") - pl.col("imb_first")).alias("imb_disp"),
        (pl.col("mid_std") / (pl.col("mid_first").abs() + 1e-12)).alias("mid_vol_rel"),
        (pl.col("mid_max") - pl.col("mid_min")).alias("mid_range"),
    )

    # segment stats: need per-segment aggregates via ntile over row_number
    seg_agg = (
        lf.with_columns(
            pl.int_range(0, pl.len()).over("sample_id").alias("_rn")
        )
        .with_columns((pl.col("_rn") / pl.len().over("sample_id")).alias("_frac"))
        .filter(pl.col("_frac") < 1.0)
        .with_columns(
            pl.when(pl.col("_frac") < 0.333).then(pl.lit("early"))
            .when(pl.col("_frac") < 0.667).then(pl.lit("mid"))
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
        .pivot(index="sample_id", on="_seg", values=[
            "mid_seg_mean", "mid_seg_std", "spread_seg_mean", "depth_seg_mean", "imb_seg_mean",
        ])
    )

    result = n_snap.join(whole, on="sample_id", how="left").join(seg_agg, on="sample_id", how="left")

    # recent/early evolution ratios
    result = result.with_columns(
        (pl.col("mid_seg_std_recent") / (pl.col("mid_seg_std_early") + 1e-12)).alias("vol_recent_early"),
        (pl.col("spread_seg_mean_recent") / (pl.col("spread_seg_mean_early") + 1e-12)).alias("spread_recent_early"),
        (pl.col("depth_seg_mean_recent") / (pl.col("depth_seg_mean_early") + 1e-12)).alias("depth_recent_early"),
        (pl.col("imb_seg_mean_recent") - pl.col("imb_seg_mean_early")).alias("imb_recent_early"),
        (pl.col("mid_seg_mean_recent") - pl.col("mid_seg_mean_early")).alias("mid_path_drift"),
    ).sort("sample_id")

    names = [c for c in result.columns if c != "sample_id"]
    vals = result.select(names).to_numpy().astype(np.float64)
    ids = result["sample_id"].to_numpy(dtype=np.int64)
    return ids, vals, names


def main() -> None:
    ids, vals, names = build_path_descriptors()
    print(f"descriptors: {vals.shape} ({len(names)} cols)")
    df = pl.DataFrame({"sample_id": ids}).with_columns(
        [pl.Series(n, vals[:, i]) for i, n in enumerate(names)]
    )
    df.write_parquet(DESC_PATH)
    print(f"saved -> {DESC_PATH}")


if __name__ == "__main__":
    main()
