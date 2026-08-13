# -*- coding: utf-8 -*-
"""P4-03: Target / Normalization Mechanism Forensics.

No models. Questions:
1. What does target look like? (distribution, monthly structure, tails)
2. Is target a return-like quantity? (|y| scales with volatility/activity)
3. Is there a normalization reference we lost? (group-mean structure,
   demean evidence, level dependence, 0.5-price-cluster behavior)
4. How much of target is a linear function of window-observable proxies?
   (if R2 is high -> target is mostly window information; if ~0 -> target
   carries future-only information)
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import polars as pl

MARKET = Path(r"D:\mscapital-forecasting\data\raw\train\market.feather")
LABEL = Path(r"D:\mscapital-forecasting\data\raw\train\label.feather")
ORDER = Path(r"D:\kaggle\working\processed_data\train_order_secondly.feather")
TX = Path(r"D:\kaggle\working\processed_data\train_transaction_secondly.feather")
OUT = Path(r"D:\mscapital-kaggle\output\p4_03_target_forensics")
OUT.mkdir(parents=True, exist_ok=True)


def main() -> None:
    lab = pl.read_ipc(LABEL).sort("sample_id")
    y = lab["target"].to_numpy().astype(np.float64)
    month = lab["month"].to_numpy()
    sid = lab["sample_id"].to_numpy()
    n = len(y)
    print(f"n={n} y: mean={y.mean():+.6e} std={y.std():.6f} "
          f"p1={np.quantile(y, .01):.6f} p99={np.quantile(y, .99):.6f} "
          f"min={y.min():.6f} max={y.max():.6f}")

    # ---- 1. monthly structure ----
    print("\n=== monthly mean/std (normalization check) ===")
    m_means, m_stds, m_n = [], [], []
    for m in range(71):
        mask = month == m
        m_means.append(y[mask].mean()); m_stds.append(y[mask].std()); m_n.append(mask.sum())
    m_means = np.asarray(m_means); m_stds = np.asarray(m_stds); m_n = np.asarray(m_n)
    print(f"monthly mean: min={m_means.min():+.6f} max={m_means.max():+.6f} std_of_means={m_means.std():.6f}")
    print(f"monthly std:  min={m_stds.min():.6f} max={m_stds.max():.6f} ratio={m_stds.max()/m_stds.min():.2f}")
    print(f"|mean|/se per month max: {(np.abs(m_means)/ (m_stds/np.sqrt(m_n))).max():.2f} "
          f"(>3 意味着月均值显著非零)")

    # ---- 2. window proxies from market (60s part) ----
    print("\n=== building window proxies (market 60s) ===")
    mkt = pl.read_ipc(MARKET, columns=[
        "sample_id", "seconds_before_predict", "ask_price_1", "bid_price_1",
        "ask_volume_1", "bid_volume_1", "transaction_volume", "transaction_count",
    ])
    mkt = mkt.with_columns([
        ((pl.col("ask_price_1") + pl.col("bid_price_1")) / 2).alias("mid"),
        (pl.col("ask_price_1") - pl.col("bid_price_1")).alias("spread"),
        (pl.col("bid_volume_1") + pl.col("ask_volume_1")).alias("depth"),
    ]).filter(pl.col("seconds_before_predict") < 60)
    win = mkt.group_by("sample_id").agg(
        pl.col("mid").first().alias("mid_first"),
        pl.col("mid").last().alias("mid_last"),
        pl.col("mid").std().alias("mid_std"),
        pl.col("spread").mean().alias("spread_mean"),
        pl.col("depth").mean().alias("depth_mean"),
        pl.col("transaction_volume").sum().alias("txvol"),
        pl.col("transaction_count").sum().alias("txcnt"),
        pl.len().alias("n_snap"),
    ).sort("sample_id")
    w = win.join(pl.DataFrame({"sample_id": sid}), on="sample_id").sort("sample_id")
    pos = np.searchsorted(win["sample_id"].to_numpy(), sid)
    w = win[pos]
    ret_w = (w["mid_last"].to_numpy() - w["mid_first"].to_numpy()) / (w["mid_first"].to_numpy() + 1e-12)
    proxies = {
        "ret_window": ret_w,
        "mid_std": w["mid_std"].to_numpy(),
        "spread_mean": w["spread_mean"].to_numpy(),
        "depth_mean": w["depth_mean"].to_numpy(),
        "txvol": w["txvol"].to_numpy(),
        "txcnt": w["txcnt"].to_numpy(),
        "n_snap": w["n_snap"].to_numpy(),
    }
    print("\n=== corr(target, window proxies) ===")
    for name, v in proxies.items():
        mask = np.isfinite(v) & np.isfinite(y)
        c = np.corrcoef(v[mask], y[mask])[0, 1]
        print(f"  corr(y, {name:12s}) = {c:+.4f}")
    # |y| vs volatility/activity (return-scaling fingerprint)
    print("\n=== |y| scaling (return-like fingerprint) ===")
    for name, v in proxies.items():
        mask = np.isfinite(v) & np.isfinite(y)
        c = np.corrcoef(np.abs(y[mask]), v[mask])[0, 1]
        print(f"  corr(|y|, {name:12s}) = {c:+.4f}")
    # binned: |y| by volatility decile
    mv = proxies["mid_std"]
    mask = np.isfinite(mv) & np.isfinite(y)
    dec = np.quantile(mv[mask], np.linspace(0, 1, 11))
    print("  |y| by mid_std decile:", end="")
    for k in range(10):
        lo_, hi_ = dec[k], dec[k + 1]
        m2 = mask & (mv >= lo_) & (mv <= hi_)
        if m2.sum() > 0:
            print(f" {np.abs(y[m2]).mean():.5f}", end="")
    print()

    # ---- 3. linear explainability: y ~ proxies ----
    print("\n=== R2 of y on window proxies (linear) ===")
    X = np.column_stack([np.nan_to_num(proxies[k], nan=0.0) for k in proxies])
    X = np.column_stack([X, np.ones(n)])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    yhat = X @ beta
    ss_res = np.sum((y - yhat) ** 2); ss_tot = np.sum((y - y.mean()) ** 2)
    print(f"  R2 = {1 - ss_res / ss_tot:.6f}  (near 0 -> target is future-dominated)")

    # ---- 4. 0.5-price cluster target behavior ----
    mkt2 = pl.read_ipc(MARKET, columns=["sample_id", "ask_price_1", "bid_price_1"])
    med = mkt2.with_columns(((pl.col("ask_price_1") + pl.col("bid_price_1")) / 2).alias("mid")) \
        .group_by("sample_id").agg(pl.col("mid").median().alias("mid_med"))
    low = med.filter(pl.col("mid_med") < 0.7)["sample_id"].to_numpy()
    low_mask = np.isin(sid, low)
    print(f"\n=== 0.5-cluster: {low_mask.sum()} samples ===")
    print(f"  y mean: low={y[low_mask].mean():+.6f} vs all={y.mean():+.6f}")
    print(f"  y std:  low={y[low_mask].std():.6f} vs all={y.std():.6f}")
    print(f"  |y| ratio: {np.abs(y[low_mask]).mean()/np.abs(y).mean():.3f}")

    np.savez_compressed(OUT / "target_forensics.npz",
                        y=y, month=month, m_means=m_means, m_stds=m_stds,
                        ret_w=ret_w, mid_std=proxies["mid_std"],
                        low_mask=low_mask)
    (OUT / "report.md").write_text("\n".join([
        "# P4-03 Target forensics", "",
        f"- y: mean={y.mean():+.2e} std={y.std():.6f} range=[{y.min():.4f},{y.max():.4f}]",
        f"- monthly mean std={m_means.std():.6f} (max |mean|/se={np.abs(m_means).max()/(m_stds/np.sqrt(m_n)).max():.2f})",
        f"- monthly std ratio={m_stds.max()/m_stds.min():.2f}",
        f"- corr(y, ret_window)={np.corrcoef(ret_w, y)[0,1]:+.4f}",
        f"- corr(|y|, mid_std)={np.corrcoef(np.abs(y), np.nan_to_num(proxies['mid_std']))[0,1]:+.4f}",
        f"- R2(y ~ proxies)={1 - ss_res/ss_tot:.6f}",
        f"- 0.5-cluster: n={low_mask.sum()}, y_std={y[low_mask].std():.6f} vs {y.std():.6f}", "",
    ]), encoding="utf-8")
    print("\nwritten to", OUT)


if __name__ == "__main__":
    main()
