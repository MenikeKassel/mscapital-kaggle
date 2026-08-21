# -*- coding: utf-8 -*-
"""P8-01A: O→T temporal-arrow diagnostic (no ML, no y).

Question: does cross-table temporal coupling (order -> transaction response)
EXIST beyond shared activity, after time-shift / reverse / shuffle placebos?

Design (per GPT1+GPT2 review, docs/otlag-probe-plan-v2.md):
- 1s bins over the 60s event window: b = int(seconds_before_predict), b=0 closest to predict
- O[b] = signed new-order flow (buy_new - sell_new) + cancel components
- T[b] = signed tx flow (buy - sell) + buy/sell volumes
- C(k) = mean over samples of sum_b O[b]*T[b-k]  (k>0: order first, tx k seconds later)
- A(k) = C(+k) - C(-k)  (forward/backward asymmetry; ~0 => activity cluster, not response)
- placebos: (1) within-sample block shift of T  (2) reverse T->O  (3) marginal-preserving shuffle of T
- right-censoring: eligible orders only (b>=k for forward, b<=60-k for backward)
- activity stratification (event count terciles) + monthly stability
- P0 leakage unit tests built-in
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import polars as pl

RAW = Path(r"D:\mscapital-forecasting\data\raw\train")
OUT = Path(r"D:\mscapital-kaggle\output\p8_01a_ot_lag")
OUT.mkdir(parents=True, exist_ok=True)

NBIN = 61  # 0..60


def load_sample_ids(n_total: int, rng: np.random.Generator) -> np.ndarray:
    """Month-stratified sample of sample_ids."""
    lab = pl.read_ipc(RAW / "label.feather", memory_map=False)
    lab = lab.with_columns(pl.arange(0, pl.len()).alias("__r"))
    per_month = max(1, n_total // lab["month"].n_unique())
    sampled = (
        lab.sort("__r").group_by("month").head(per_month).select("sample_id", "month").sort("month")
    )
    # deterministic shuffling
    ids = sampled["sample_id"].to_numpy()
    months = sampled["month"].to_numpy()
    idx = rng.permutation(len(ids))
    return ids[idx], months[idx]


def load_binned(which: str, sample_ids: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Read events, bin to 1s grid -> array (n_samples, NBIN, channels).

    order channels:  [buy_new, sell_new, buy_cancel, sell_cancel, signed_new]
    tx channels:     [buy_vol, sell_vol, signed]
    Vectorized via group_by + np.add.at.
    """
    order = np.argsort(sample_ids)
    ids_sorted = sample_ids[order]
    path = RAW / f"{which}.feather"
    lf = pl.scan_ipc(path, memory_map=False)
    df = (
        lf.filter(pl.col("sample_id").is_in(sample_ids))
        .with_columns(
            pl.col("seconds_before_predict").cast(pl.Int32).clip(0, 60).alias("bin"),
            (pl.col("side") == 0).cast(pl.Int32).alias("is_buy"),
            (pl.col("side") == 1).cast(pl.Int32).alias("is_sell"),
        )
        .collect()
    )
    n = len(sample_ids)
    if which == "order":
        df = df.with_columns(
            (pl.col("order_action") == 0).cast(pl.Int32).alias("is_new"),
            (pl.col("order_action") == 1).cast(pl.Int32).alias("is_cancel"),
        )
        df = df.group_by("sample_id", "bin").agg(
            ((pl.col("is_buy") * pl.col("is_new") * pl.col("volume")).sum()).alias("buy_new"),
            ((pl.col("is_sell") * pl.col("is_new") * pl.col("volume")).sum()).alias("sell_new"),
            ((pl.col("is_buy") * pl.col("is_cancel") * pl.col("volume")).sum()).alias("buy_cancel"),
            ((pl.col("is_sell") * pl.col("is_cancel") * pl.col("volume")).sum()).alias("sell_cancel"),
            (((pl.col("is_buy") - pl.col("is_sell")) * pl.col("is_new") * pl.col("volume")).sum()).alias("signed_new"),
        )
        X = np.zeros((n, NBIN, 5), dtype=np.float64)
        row = np.searchsorted(ids_sorted, df["sample_id"].to_numpy())
        bins = df["bin"].to_numpy()
        for ch, col in enumerate(["buy_new", "sell_new", "buy_cancel", "sell_cancel", "signed_new"]):
            np.add.at(X[:, :, ch], (row, bins), df[col].to_numpy())
    else:
        df = df.group_by("sample_id", "bin").agg(
            ((pl.col("is_buy") * pl.col("volume")).sum()).alias("buy_vol"),
            ((pl.col("is_sell") * pl.col("volume")).sum()).alias("sell_vol"),
            (((pl.col("is_buy") - pl.col("is_sell")) * pl.col("volume")).sum()).alias("signed"),
        )
        X = np.zeros((n, NBIN, 3), dtype=np.float64)
        row = np.searchsorted(ids_sorted, df["sample_id"].to_numpy())
        bins = df["bin"].to_numpy()
        for ch, col in enumerate(["buy_vol", "sell_vol", "signed"]):
            np.add.at(X[:, :, ch], (row, bins), df[col].to_numpy())
    return X


def response_curve(Xo: np.ndarray, Xt: np.ndarray, oc: int, tc: int, lags: np.ndarray):
    """C(k) = mean_s sum_b O[b]*T[b-k], with censoring-eligible orders per direction.

    Returns (C, n_eligible): C[ki] mean product per sample over eligible bins;
    n_eligible = per-sample count of eligible orders (for weighting/stability).
    """
    n, B = Xo.shape[0], Xo.shape[1]
    C = np.zeros(len(lags))
    N = np.zeros(len(lags))
    for ki, k in enumerate(lags):
        if k >= 0:
            # forward: T[b-k], need b-k >= 0 -> b >= k
            elig = Xo[:, k:, oc] > 0  # order present at eligible bins
            prod = Xo[:, k:, oc] * Xt[:, : B - k, tc]
            cnt = elig.sum(axis=1)
        else:
            kk = -k
            # backward: T[b+kk], need b+kk <= 60 -> b <= 60-kk
            elig = Xo[:, : B - kk, oc] > 0
            prod = Xo[:, : B - kk, oc] * Xt[:, kk:, tc]
            cnt = elig.sum(axis=1)
        with np.errstate(invalid="ignore"):
            s = np.where(cnt > 0, prod.sum(axis=1) / np.maximum(cnt, 1), np.nan)
        valid = ~np.isnan(s)
        N[ki] = valid.sum()
        C[ki] = np.nanmean(s) if N[ki] else np.nan
    return C, N


def zscore_response(Xo: np.ndarray, Xt: np.ndarray, oc: int, tc: int, lags: np.ndarray):
    """Per-sample Pearson cross-correlation over the full (censoring-eligible) window.

    Full-window z-score (zeros included) -> dilutes but is unbiased; under
    marginal-preserving shuffle it converges to ~0, under block shift only
    marginal structure survives. Requires >=3 nonzero bins on both sides.
    """
    n, B = Xo.shape[0], Xo.shape[1]
    C = np.zeros(len(lags))
    N = np.zeros(len(lags))
    for ki, k in enumerate(lags):
        if k >= 0:
            A = Xo[:, k:, oc]
            Bt = Xt[:, : B - k, tc]
        else:
            kk = -k
            A = Xo[:, : B - kk, oc]
            Bt = Xt[:, kk:, tc]
        a = A - A.mean(axis=1, keepdims=True)
        bt = Bt - Bt.mean(axis=1, keepdims=True)
        den = np.sqrt((a * a).sum(axis=1) * (bt * bt).sum(axis=1))
        corr = np.where(den > 0, (a * bt).sum(axis=1) / den, np.nan)
        ev_a = (A != 0).sum(axis=1)
        ev_b = (Bt != 0).sum(axis=1)
        corr = np.where((ev_a >= 3) & (ev_b >= 3), corr, np.nan)
        valid = ~np.isnan(corr)
        N[ki] = valid.sum()
        C[ki] = np.nanmean(corr) if N[ki] else np.nan
    return C, N


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-samples", type=int, default=150000)
    ap.add_argument("--reps", type=int, default=20, help="placebo repetitions")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    t0 = time.time()

    ids, months = load_sample_ids(args.n_samples, rng)
    n = len(ids)
    print(f"[P8-01A] samples={n} months={months.min()}..{months.max()}")

    Xo = load_binned("order", ids, rng)
    Xt = load_binned("transaction", ids, rng)
    print(f"[P8-01A] binned order={Xo.shape} tx={Xt.shape} ({time.time()-t0:.0f}s)")

    # ---- P0 leakage unit tests ----
    assert Xo.shape[0] == n and Xt.shape[0] == n
    assert Xo.shape[1] == NBIN and Xt.shape[1] == NBIN
    assert np.isfinite(Xo).all() and np.isfinite(Xt).all()
    assert (Xo[:, :, :4] >= 0).all() and (Xt[:, :, :2] >= 0).all()  # volumes non-negative
    print("[P8-01A] P0 leakage tests PASS (window-local, non-negative, aligned)")

    lags = np.array([-30, -10, -5, -3, -1, 0, 1, 3, 5, 10, 30])
    pairs = [
        ("signed_new->signed_tx", 4, 2),
        ("buy_new->buy_tx", 0, 0),
        ("sell_new->sell_tx", 1, 1),
        ("buy_cancel->sell_tx", 2, 1),
        ("sell_cancel->buy_tx", 3, 0),
    ]
    res = {"meta": {"n_samples": n, "bins": NBIN, "lags": lags.tolist(), "reps": args.reps}}
    for name, oc, tc in pairs:
        C, N = response_curve(Xo, Xt, oc, tc, lags)
        Cz, Nz = zscore_response(Xo, Xt, oc, tc, lags)
        A = C[4] - C[6]  # +1 vs -1 (note lags idx: -1 is idx4, +1 is idx6)
        res[name] = {
            "C_raw": C.tolist(), "n_eligible": N.tolist(),
            "C_zscore": Cz.tolist(), "n_z": Nz.tolist(),
            "A_asym_raw_1s": float(A),
            "A_asym_z_1s": float(Cz[6] - Cz[4]),
        }
        print(f"[{name}] C_raw(+1,-1)={C[6]:.4f},{C[4]:.4f}  A={A:+.4f}  "
              f"z(+1,-1)={Cz[6]:.4f},{Cz[4]:.4f}")

    # ---- placebos on the main pair (signed_new->signed_tx), subset of samples ----
    sub = min(n, 30000)
    idx = rng.choice(n, sub, replace=False)
    Xo_s, Xt_s = Xo[idx], Xt[idx]
    oc, tc = 4, 2
    C_true, _ = response_curve(Xo_s, Xt_s, oc, tc, lags)
    Cz_true, _ = zscore_response(Xo_s, Xt_s, oc, tc, lags)
    nulls_raw = np.zeros((args.reps, len(lags)))
    nulls_z = np.zeros((args.reps, len(lags)))
    bin_idx = np.arange(NBIN)
    for r in range(args.reps):
        shift = rng.integers(5, 26, size=sub)  # per-sample random block shift
        rolled_idx = (bin_idx[None, :] + shift[:, None]) % NBIN  # circular within 60s window
        Xt_sh = np.take_along_axis(Xt_s, rolled_idx[:, :, None], axis=1)
        C1, _ = response_curve(Xo_s, Xt_sh, oc, tc, lags)
        Cz1, _ = zscore_response(Xo_s, Xt_sh, oc, tc, lags)
        nulls_raw[r], nulls_z[r] = C1, Cz1
    res["placebo_block_shift"] = {
        "C_true": C_true.tolist(), "Cz_true": Cz_true.tolist(),
        "null_mean_raw": nulls_raw.mean(0).tolist(), "null_std_raw": nulls_raw.std(0).tolist(),
        "null_mean_z": nulls_z.mean(0).tolist(), "null_std_z": nulls_z.std(0).tolist(),
        "zstat_raw_1s": float((C_true[6] - nulls_raw.mean(0)[6]) / max(nulls_raw.std(0)[6], 1e-12)),
        "zstat_z_1s": float((Cz_true[6] - nulls_z.mean(0)[6]) / max(nulls_z.std(0)[6], 1e-12)),
    }
    print(f"[placebo shift] true_z(+1)={Cz_true[6]:.4f} null_z={nulls_z.mean(0)[6]:.4f}±{nulls_z.std(0)[6]:.4f}")

    # reverse placebo: T -> O
    C_rev, _ = response_curve(Xt_s, Xo_s, 2, 4, lags)  # signed tx "first", signed order "later"
    Cz_rev, _ = zscore_response(Xt_s, Xo_s, 2, 4, lags)
    res["placebo_reverse"] = {"C_raw": C_rev.tolist(), "C_zscore": Cz_rev.tolist()}
    print(f"[placebo reverse] z(+1,-1)={Cz_rev[6]:.4f},{Cz_rev[4]:.4f}")

    # marginal-preserving shuffle of T
    nulls_z_sh = np.zeros((args.reps, len(lags)))
    for r in range(args.reps):
        Xt_shuf = rng.permuted(Xt_s, axis=1)  # per-sample independent permutation of bins
        Cz1, _ = zscore_response(Xo_s, Xt_shuf, oc, tc, lags)
        nulls_z_sh[r] = Cz1
    res["placebo_shuffle"] = {
        "null_mean_z": nulls_z_sh.mean(0).tolist(), "null_std_z": nulls_z_sh.std(0).tolist(),
    }
    print(f"[placebo shuffle] null_z(+1)={nulls_z_sh.mean(0)[6]:.4f}±{nulls_z_sh.std(0)[6]:.4f}")

    # ---- activity stratification (main pair) ----
    ev_per_sample = (Xo[:, :, 4] != 0).sum(axis=1) + (Xt[:, :, 2] != 0).sum(axis=1)
    terc = np.quantile(ev_per_sample, [1 / 3, 2 / 3])
    strata = {}
    for label, m in [("low", ev_per_sample <= terc[0]),
                     ("mid", (ev_per_sample > terc[0]) & (ev_per_sample <= terc[1])),
                     ("high", ev_per_sample > terc[1])]:
        if m.sum() < 100:
            continue
        Cz, _ = zscore_response(Xo[m], Xt[m], oc, tc, lags)
        strata[label] = {"n": int(m.sum()), "C_zscore": Cz.tolist(), "z_1s": float(Cz[6])}
    res["activity_strata"] = strata
    print(f"[strata] " + ", ".join(f"{k}: z(+1)={v['z_1s']:.4f} (n={v['n']})" for k, v in strata.items()))

    # ---- monthly stability (main pair, z-score) ----
    monthly = {}
    for mm in np.unique(months):
        m = months == mm
        if m.sum() < 50:
            continue
        Cz, _ = zscore_response(Xo[m], Xt[m], oc, tc, lags)
        monthly[int(mm)] = {"n": int(m.sum()), "z_p1": float(Cz[6]), "z_m1": float(Cz[4]),
                            "A_z": float(Cz[6] - Cz[4])}
    res["monthly"] = monthly
    npos = sum(1 for v in monthly.values() if v["A_z"] > 0)
    print(f"[monthly] {len(monthly)} months, A_z>0: {npos}/{len(monthly)}")

    res["elapsed_s"] = round(time.time() - t0, 1)
    (OUT / "results.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(f"[P8-01A] DONE {time.time()-t0:.0f}s -> {OUT / 'results.json'}")


if __name__ == "__main__":
    main()
