# -*- coding: utf-8 -*-
"""P4-01(a)-0 Data Integrity: schema / NaN / sample length / time coverage / duplicates.

Runs on test/market.feather (the split that aligns with |d|).
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import polars as pl

from p4_01a_common import MARKET_TEST, OUT, load_d

COLS = [
    "transaction_avgprice", "transaction_volume", "transaction_count",
    "ask_price_1", "ask_volume_1", "bid_price_1", "bid_volume_1",
    "ask_price_2", "ask_volume_2", "bid_price_2", "bid_volume_2",
]


def main() -> None:
    lf = pl.scan_ipc(MARKET_TEST)
    total = lf.select(pl.len()).collect().item()
    print(f"=== P4-01(a)-0 Data Integrity (test/market.feather) ===")
    print(f"total rows: {total:,}")

    # 1. schema
    schema = lf.collect_schema()
    print(f"\n[1] schema ({len(schema)} cols):")
    for k, v in schema.items():
        print(f"    {k}: {v}")

    # 2. NaN per column (scan-based)
    nan_exprs = [pl.col(c).is_null().sum().alias(c) for c in COLS]
    nan_df = lf.select(nan_exprs).collect()
    print(f"\n[2] NaN counts:")
    for c in COLS:
        n = nan_df[c].item()
        print(f"    {c}: {n:,} ({n / total * 100:.2f}%)")

    # 3. sample length distribution
    lens = lf.group_by("sample_id").agg(pl.len().alias("n")).collect()
    print(f"\n[3] sample length:")
    print(f"    samples: {lens.height:,}")
    print(f"    min={lens['n'].min()} p10={lens['n'].quantile(0.10):.0f} "
          f"median={lens['n'].median():.0f} p90={lens['n'].quantile(0.90):.0f} max={lens['n'].max()}")

    # 4. time coverage
    t = lf.select(
        pl.col("seconds_before_predict").min().alias("min"),
        pl.col("seconds_before_predict").max().alias("max"),
        pl.col("seconds_before_predict").mean().alias("mean"),
        pl.col("seconds_before_predict").std().alias("std"),
    ).collect()
    print(f"\n[4] seconds_before_predict: min={t['min'].item():.3f} max={t['max'].item():.3f} "
          f"mean={t['mean'].item():.3f} std={t['std'].item():.3f}")

    # unique seconds per sample (typical spacing)
    uniq = lf.select(pl.col("seconds_before_predict").n_unique().alias("n")).collect().item()
    print(f"    distinct timestamps overall: {uniq:,}")

    # 5. duplicates on (sample_id, seconds_before_predict)
    dup = lf.group_by(["sample_id", "seconds_before_predict"]).agg(pl.len().alias("n")).collect()
    max_dup = dup["n"].max()
    dup_ratio = dup.height / total
    print(f"\n[5] (sample_id, seconds) duplicate check:")
    print(f"    unique pairs: {dup.height:,} / {total:,} (ratio {dup_ratio:.6f})")
    print(f"    max count of one pair: {max_dup}")

    # 6. sample_id range / d alignment
    ids, d, ad = load_d()
    print(f"\n[6] d alignment: d_analysis has {ids.shape[0]:,} samples, "
          f"market has {lens.height:,} samples")
    overlap = np.intersect1d(ids, lens["sample_id"].to_numpy().astype(np.int64))
    print(f"    overlap: {overlap.shape[0]:,}")

    report = {
        "total_rows": int(total),
        "n_samples": int(lens.height),
        "n_d_samples": int(ids.shape[0]),
        "n_overlap": int(overlap.shape[0]),
        "nan_counts": {c: int(nan_df[c].item()) for c in COLS},
        "len_min": int(lens["n"].min()), "len_median": int(lens["n"].median()),
        "len_max": int(lens["n"].max()),
        "time_min": float(t["min"].item()), "time_max": float(t["max"].item()),
        "dup_ratio": float(dup_ratio), "dup_max": int(max_dup),
    }
    np.save(OUT / "integrity.npy", report, allow_pickle=True)
    print(f"\nsaved -> {OUT / 'integrity.npy'}")


if __name__ == "__main__":
    main()
