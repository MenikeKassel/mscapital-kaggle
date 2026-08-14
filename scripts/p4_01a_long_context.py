# -*- coding: utf-8 -*-
"""P4-01a: 600s long-context information test + 0.5-group forensic (P4-04b).

Core question: does the 540s of market history BEFORE the last 60s explain
d_orth = p_ref - E[p_ref | p_v7]? If M_long (only -600..-60s) explains d_orth
beyond M_short (only last 60s), LB142 genuinely uses long-horizon context.

Segments (seconds_before_predict): [300,600)=[-600,-300), [120,300)=[-300,-120),
[60,120)=[-120,-60), [0,60)=[-60,0].
Per segment: mid_std, spread_mean, depth_mean, tx_vol, mid_trend, jump_count,
autocorr_lag1.

P4-04b (attached): 0.5-price cluster - month distribution, activity,
target structure (train), and test-side existence + |d| relation.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import polars as pl

REF = Path(r"D:\mscapital-forecasting\reference\lb142\submission_ref_lb0142.csv")
V7 = Path(r"D:\mscapital-kaggle\output\submissions\submission_blend_v7_rl20.csv")
MARKET_TRAIN = Path(r"D:\mscapital-forecasting\data\raw\train\market.feather")
MARKET_TEST = Path(r"D:\mscapital-forecasting\data\raw\test\market.feather")
LABEL = Path(r"D:\mscapital-forecasting\data\raw\train\label.feather")
OUT = Path(r"D:\mscapital-kaggle\output\p4_01a_long_context")
OUT.mkdir(parents=True, exist_ok=True)

SEGMENTS = [(300, 600), (120, 300), (60, 120), (0, 60)]
SEG_NAMES = ["m600_300", "m300_120", "m120_60", "m60_0"]


def segment_features(market_path: Path) -> np.ndarray:
    """Return (n_samples, 4 segments x 7 feats) aligned to sample_id order."""
    mkt = pl.read_ipc(market_path, columns=[
        "sample_id", "seconds_before_predict", "ask_price_1", "bid_price_1",
        "ask_volume_1", "bid_volume_1", "transaction_volume",
    ])
    mkt = mkt.with_columns([
        ((pl.col("ask_price_1") + pl.col("bid_price_1")) / 2).alias("mid"),
        (pl.col("ask_price_1") - pl.col("bid_price_1")).alias("spread"),
        (pl.col("bid_volume_1") + pl.col("ask_volume_1")).alias("depth"),
    ])
    n = int(mkt["sample_id"].max()) + 1
    X = np.zeros((n, 4, 7), dtype=np.float32)
    for si, (s0, s1) in enumerate(SEGMENTS):
        seg = mkt.filter((pl.col("seconds_before_predict") >= s0) & (pl.col("seconds_before_predict") < s1))
        if seg.height == 0:
            continue
        g = seg.sort("sample_id", "seconds_before_predict").group_by("sample_id").agg(
            pl.col("mid").std().alias("mid_std"),
            pl.col("spread").mean().alias("spread"),
            pl.col("depth").mean().alias("depth"),
            pl.col("transaction_volume").sum().alias("txvol"),
            pl.col("mid").first().alias("mid_f"),
            pl.col("mid").last().alias("mid_l"),
            pl.col("mid").diff().abs().mean().alias("jump"),
            pl.corr(pl.col("mid"), pl.col("mid").shift(1)).alias("ac"),
        )
        ids = g["sample_id"].to_numpy()
        X[ids, si, 0] = np.nan_to_num(g["mid_std"].to_numpy(), nan=0.0)
        X[ids, si, 1] = np.nan_to_num(g["spread"].to_numpy(), nan=0.0)
        X[ids, si, 2] = np.nan_to_num(g["depth"].to_numpy(), nan=0.0)
        X[ids, si, 3] = np.nan_to_num(g["txvol"].to_numpy(), nan=0.0)
        X[ids, si, 4] = np.nan_to_num(
            ((g["mid_l"].to_numpy() - g["mid_f"].to_numpy()) / (g["mid_f"].to_numpy() + 1e-12)), nan=0.0)
        X[ids, si, 5] = np.nan_to_num(g["jump"].to_numpy(), nan=0.0)
        X[ids, si, 6] = np.nan_to_num(g["ac"].to_numpy(), nan=0.0)
    return X


def main() -> None:
    ref = pl.read_csv(REF).sort("sample_id")
    v7 = pl.read_csv(V7).sort("sample_id")
    ids = ref["sample_id"].to_numpy()
    p_ref = ref["prediction"].to_numpy().astype(np.float64)
    p_v7 = v7["prediction"].to_numpy().astype(np.float64)
    # d_orth: residual of ref on v7
    A = np.column_stack([p_v7, np.ones(len(p_v7))])
    beta, *_ = np.linalg.lstsq(A, p_ref, rcond=None)
    d_orth = p_ref - A @ beta
    print(f"corr(ref, v7)={np.corrcoef(p_ref, p_v7)[0,1]:.4f} d_orth std={d_orth.std():.6f}")

    print("building test market segment features...")
    X = segment_features(MARKET_TEST)
    pos = np.searchsorted(np.arange(X.shape[0]), ids)
    X = X[ids]
    print(f"X: {X.shape}")

    flat = X.reshape(X.shape[0], -1)
    # NaN-safe
    flat = np.nan_to_num(flat, nan=0.0)
    n_seg = len(SEGMENTS)
    seg_names_flat = [f"{SEG_NAMES[s]}_{k}" for s in range(n_seg) for k in range(7)]

    def r2(Xm, y):
        Xm = np.column_stack([Xm, np.ones(len(y))])
        b, *_ = np.linalg.lstsq(Xm, y, rcond=None)
        yhat = Xm @ b
        return 1 - np.sum((y - yhat) ** 2) / np.sum((y - y.mean()) ** 2)

    short_idx = [i for i in range(flat.shape[1]) if SEG_NAMES[3] in seg_names_flat[i]]
    long_idx = [i for i in range(flat.shape[1]) if SEG_NAMES[3] not in seg_names_flat[i]]
    r2_short = r2(flat[:, short_idx], d_orth)
    r2_long = r2(flat[:, long_idx], d_orth)
    r2_both = r2(flat, d_orth)
    print(f"\n=== R2 on d_orth (test, n={len(d_orth)}) ===")
    print(f"  M_short (last 60s):   R2 = {r2_short:.6f}")
    print(f"  M_long  (-600..-60s): R2 = {r2_long:.6f}")
    print(f"  M_short + M_long:     R2 = {r2_both:.6f}")
    print(f"  long increment:       {r2_both - r2_short:.6f}")

    # permutation null for long-only R2
    rng = np.random.default_rng(2026)
    null = []
    for _ in range(10):
        perm = rng.permutation(d_orth)
        null.append(r2(flat[:, long_idx], perm))
    null = np.asarray(null)
    print(f"  permutation null (10x): mean={null.mean():.6f} max={null.max():.6f}")
    print(f"  long R2 vs null max: ratio={r2_long / max(null.max(), 1e-12):.1f}x")

    # segment-level: which segment explains d_orth?
    print("\n=== per-segment R2 on d_orth ===")
    for s in range(n_seg):
        idx = [i for i in range(flat.shape[1]) if seg_names_flat[i].startswith(SEG_NAMES[s])]
        print(f"  {SEG_NAMES[s]:10s} R2 = {r2(flat[:, idx], d_orth):.6f}")

    # ---------- P4-04b: 0.5 group forensic (train) ----------
    print("\n=== P4-04b 0.5-group forensic (train) ===")
    mkt_tr = pl.read_ipc(MARKET_TRAIN, columns=["sample_id", "seconds_before_predict", "ask_price_1", "bid_price_1"])
    med = mkt_tr.with_columns(((pl.col("ask_price_1") + pl.col("bid_price_1")) / 2).alias("mid")) \
        .group_by("sample_id").agg(pl.col("mid").median().alias("mid_med"))
    low_ids = med.filter(pl.col("mid_med") < 0.7)["sample_id"].to_numpy()
    print(f"low group: n={len(low_ids)}")
    lab = pl.read_ipc(LABEL)
    lab_low = lab.filter(pl.col("sample_id").is_in(low_ids))
    mo = lab_low.group_by("month").len().sort("month")
    months_present = mo["month"].to_numpy()
    print(f"months present: {months_present.min()}..{months_present.max()} (n months={len(months_present)})")
    per_month = mo["len"].to_numpy()
    print(f"per-month counts: min={per_month.min()} max={per_month.max()} mean={per_month.mean():.0f}")
    y_low = lab_low["target"].to_numpy()
    y_all = lab["target"].to_numpy()
    print(f"target: low mean={y_low.mean():+.6f} std={y_low.std():.6f} | mean|y| ratio={np.abs(y_low).mean()/np.abs(y_all).mean():.3f}")

    # test-side: does the 0.5 group exist on test? + |d| relation
    mkt_te = pl.read_ipc(MARKET_TEST, columns=["sample_id", "ask_price_1", "bid_price_1"])
    med_te = mkt_te.with_columns(((pl.col("ask_price_1") + pl.col("bid_price_1")) / 2).alias("mid")) \
        .group_by("sample_id").agg(pl.col("mid").median().alias("mid_med"))
    te_low = med_te.filter(pl.col("mid_med") < 0.7)["sample_id"].to_numpy()
    print(f"\ntest low group: n={len(te_low)} ({len(te_low)/len(ids)*100:.2f}%)")
    if len(te_low) > 0:
        low_mask = np.isin(ids, te_low)
        print(f"  |d_orth| low-group: {np.abs(d_orth[low_mask]).mean():.6f} vs all: {np.abs(d_orth).mean():.6f}")

    np.savez_compressed(OUT / "long_context.npz",
                        sample_id=ids, d_orth=d_orth, X=flat,
                        seg_names=np.array(seg_names_flat),
                        r2_short=r2_short, r2_long=r2_long, r2_both=r2_both)
    (OUT / "report.md").write_text("\n".join([
        "# P4-01a long-context test", "",
        f"- R2(M_short)={r2_short:.6f}  R2(M_long)={r2_long:.6f}  R2(both)={r2_both:.6f}",
        f"- long increment={r2_both - r2_short:.6f}", "",
    ]), encoding="utf-8")
    print("\nwritten to", OUT)


if __name__ == "__main__":
    main()
