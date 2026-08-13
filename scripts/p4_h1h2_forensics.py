# -*- coding: utf-8 -*-
"""P4 H1/H2 quick forensics: asset identity fingerprint + cross-row adjacency.

H1: price/volume/event structure across samples -- unimodal (one asset) or
    multimodal (multiple assets)? tick-size pattern? price scale clusters?
H2: sample adjacency -- does sample i's final-second price continue into
    sample j's first-second price for any ordering (sample_id order, sorted
    by price, etc.)? boundary mismatch vs random baseline.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import polars as pl

ORDER = Path(r"D:\kaggle\working\processed_data\train_order_secondly.feather")
TX = Path(r"D:\kaggle\working\processed_data\train_transaction_secondly.feather")
OUT = Path(r"D:\mscapital-kaggle\output\p4_forensics")
OUT.mkdir(parents=True, exist_ok=True)


def main() -> None:
    print("loading order/tx (sample-level stats only)...")
    order = pl.read_ipc(ORDER)
    tx = pl.read_ipc(TX)

    # ---------- H1: price structure ----------
    o_px = order["price"].to_numpy()
    t_px = tx["price"].to_numpy()
    print("\n=== H1 price structure ===")
    for name, p in (("order", o_px), ("tx", t_px)):
        print(f"{name}: n={p.size} min={p.min():.4f} max={p.max():.4f} "
              f"p1={np.quantile(p, .01):.4f} p50={np.quantile(p, .5):.4f} p99={np.quantile(p, .99):.4f}")
    # tick structure: smallest nonzero price difference within sample
    # per-sample price stats
    ps = order.group_by("sample_id").agg(
        pl.col("price").min().alias("pmin"), pl.col("price").max().alias("pmax"),
        pl.col("price").median().alias("pmed"),
    ).sort("sample_id")
    pmin = ps["pmin"].to_numpy(); pmax = ps["pmax"].to_numpy()
    pmed = ps["pmed"].to_numpy()
    print(f"per-sample price range: median range={np.median(pmax - pmin):.6f} "
          f"p99 range={np.quantile(pmax - pmin, .99):.6f}")
    print(f"per-sample median price: p1={np.quantile(pmed, .01):.4f} "
          f"p99={np.quantile(pmed, .99):.4f}  (spread of median price across samples)")
    # volume structure
    v = order.group_by("sample_id").agg(pl.col("volume").sum().alias("vsum")).sort("sample_id")
    vsum = v["vsum"].to_numpy()
    print(f"per-sample order volume: p1={np.quantile(vsum, .01):.0f} p50={np.quantile(vsum, .5):.0f} "
          f"p99={np.quantile(vsum, .99):.0f} max={vsum.max():.0f}")

    # ---------- H2: adjacency by sample_id order ----------
    print("\n=== H2 adjacency (sample_id consecutive) ===")
    # per-sample first and last second price (order side-agnostic median per second)
    o2 = order.with_columns(pl.col("seconds_before_predict").cast(pl.Int32)).filter(pl.col("price").is_not_null())
    edge = o2.filter((pl.col("seconds_before_predict") <= 2) | (pl.col("seconds_before_predict") >= 58))
    agg = edge.group_by("sample_id", "seconds_before_predict").agg(
        pl.col("price").median().alias("pmed")
    )
    first = agg.filter(pl.col("seconds_before_predict") <= 2).sort("sample_id", "seconds_before_predict")
    last = agg.filter(pl.col("seconds_before_predict") >= 58).sort("sample_id", "seconds_before_predict")
    # take min-second (earliest) and max-second (latest) per sample
    f = first.group_by("sample_id").agg(pl.col("pmed").first().alias("p_first"))
    l = last.group_by("sample_id").agg(pl.col("pmed").last().alias("p_last"))
    merged = f.join(l, on="sample_id").sort("sample_id")
    p_first = merged["p_first"].to_numpy()
    p_last = merged["p_last"].to_numpy()
    sids = merged["sample_id"].to_numpy()
    print(f"valid edge samples: {len(sids)}")
    # continuity: sample i's last price vs sample i+1's first price (sample_id order)
    jump = np.abs(p_first[1:] - p_last[:-1])
    rng = np.random.default_rng(2026)
    # random baseline: shuffle first prices
    perm = rng.permutation(p_first)
    jump_rand = np.abs(perm[1:] - p_last[:-1])
    print(f"consecutive sample_id: median |boundary jump|={np.median(jump):.6f} mean={jump.mean():.6f}")
    print(f"random pairing:        median |boundary jump|={np.median(jump_rand):.6f} mean={jump_rand.mean():.6f}")
    print(f"ratio (consecutive/random): {jump.mean() / max(jump_rand.mean(), 1e-12):.4f}")
    # within-sample continuity (control): i's first vs i's last
    within = np.abs(p_first - p_last)
    print(f"within-sample (control): median |first-last|={np.median(within):.6f}")
    np.savez_compressed(OUT / "h1_h2_forensics.npz",
                        pmin=pmin, pmax=pmax, pmed=pmed, vsum=vsum,
                        sids=sids, p_first=p_first, p_last=p_last,
                        jump=jump, jump_rand=jump_rand, within=within)


if __name__ == "__main__":
    main()
