# -*- coding: utf-8 -*-
"""P4-05: Monthly drift predictability (strict rolling-origin).

Question: can future-month target means be predicted from past months ONLY?
(Existence of month effect was already proven: max |mean|/se = 16.5.)

Design:
- mu_m = E[y | month=m] from FULL label (months 0-70)
- For each outer valid month m: mu_hat_m estimated from months < m only
- Corrections compared:
    A. no correction (baseline only)
    B. expanding historical mean of mu
    C. recent 3/6/12-month rolling mean
    D. exponentially weighted month mean (halflife 3/6/12)
    E. linear trend extrapolation (last 12 months)
- alpha grid on inner-tune months (per outer), applied to valid
- Evaluations: overall delta per outer, month-by-month delta,
  high/low activity delta (activity = market snapshot count), mean calib
- Gate: mean rolling-origin delta > 0, >=70% future months non-negative,
  no catastrophic worst block; delta >= +0.0010 is significant
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
import polars as pl

from mscapital.metrics import cosine_uncentered, normalize_prediction
from mscapital.residual import CanonicalOOF

LABEL = Path(r"D:\mscapital-forecasting\data\raw\train\label.feather")
CANONICAL = Path(r"D:\mscapital-kaggle\output\canonical_residual_oof\canonical_residual_oof.npz")
MARKET = Path(r"D:\mscapital-forecasting\data\raw\train\market.feather")
OUT = Path(r"D:\mscapital-kaggle\output\p4_05_monthly_drift")
OUT.mkdir(parents=True, exist_ok=True)

OUTER_MONTHS = {
    "PSEUDO": (21, 32),
    "H2": (21, 40),
    "T3": (21, 50),
    "T4": (21, 50),
}
INNER_TUNE = {
    "PSEUDO": (27, 32),
    "H2": (31, 40),
    "T3": (41, 50),
    "T4": (41, 50),
}
ALPHA_GRID = np.array([0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0])


def mu_estimators(mu: np.ndarray, m: int) -> dict[str, float]:
    """Predict mu_m from mu[0..m-1] only. Returns {method: estimate}."""
    hist = mu[:m]
    out: dict[str, float] = {}
    out["B_expanding"] = float(hist.mean()) if m > 0 else 0.0
    for w in (3, 6, 12):
        out[f"C_roll{w}"] = float(hist[-w:].mean()) if m >= w else (float(hist.mean()) if m > 0 else 0.0)
    for hl in (3, 6, 12):
        if m == 0:
            out[f"D_ewma{hl}"] = 0.0
        else:
            alpha = 1 - 0.5 ** (1.0 / hl)
            w = np.exp(-alpha * np.arange(m)[::-1])
            out[f"D_ewma{hl}"] = float(np.sum(w * hist) / w.sum())
    if m >= 2:
        k = min(12, m)
        x = np.arange(k, dtype=float)
        yv = hist[-k:]
        b, a = np.polyfit(x, yv, 1)
        out["E_trend"] = float(a + b * k)
    else:
        out["E_trend"] = float(hist.mean()) if m > 0 else 0.0
    return out


def main() -> None:
    lab = pl.read_ipc(LABEL)
    y_all = lab["target"].to_numpy().astype(np.float64)
    month_all = lab["month"].to_numpy()
    mu = np.array([y_all[month_all == m].mean() for m in range(71)])
    mu_std = np.array([y_all[month_all == m].std() for m in range(71)])
    n_m = np.array([(month_all == m).sum() for m in range(71)])
    print("=== mu_m diagnostics ===")
    print(f"mu: mean={mu.mean():+.6f} std={mu.std():.6f} min={mu.min():+.6f} max={mu.max():+.6f}")
    # ACF 1..12
    mu_c = mu - mu.mean()
    acf = [np.corrcoef(mu_c[:-k], mu_c[k:])[0, 1] if len(mu_c) > k else 0.0 for k in range(1, 13)]
    print("ACF(1..12):", " ".join(f"{v:+.2f}" for v in acf))
    # sign persistence
    signs = np.sign(mu_c)
    same = (signs[1:] == signs[:-1]).mean()
    print(f"sign persistence: {same:.3f} (random=0.5)")
    # trend (last 24 months linear)
    x = np.arange(24, dtype=float)
    b, a = np.polyfit(x, mu[-24:], 1)
    print(f"trend last 24m: slope={b:+.2e} (per month)")

    canonical = CanonicalOOF(**{
        k: np.asarray(np.load(CANONICAL)[k]) for k in
        ("sample_id", "month", "target", "baseline_oof", "source_train_end")
    })
    canonical.validate()
    cmonth = np.asarray(canonical.month, dtype=int)
    cy = np.asarray(canonical.target, dtype=np.float64)
    cbase = np.asarray(canonical.baseline_oof, dtype=np.float64)
    cid = np.asarray(canonical.sample_id)

    # activity: market snapshot count per sample
    print("\nloading activity (market rows/sample)...")
    act = pl.read_ipc(MARKET, columns=["sample_id"]).group_by("sample_id").len().sort("sample_id")
    a_ids = act["sample_id"].to_numpy()
    a_cnt = act["len"].to_numpy().astype(np.float64)
    apos = np.searchsorted(a_ids, cid)
    activity = a_cnt[apos]
    hi_act = activity >= np.quantile(activity, 0.9)

    # baseline normalization (RMS, per outer on the outer's rows? M01-A uses tune-fitted scale;
    # here we use global RMS scale from each outer's own rows for simplicity of the correction test)
    methods = ["B_expanding", "C_roll3", "C_roll6", "C_roll12", "D_ewma3", "D_ewma6", "D_ewma12", "E_trend"]
    summary_rows = []
    month_deltas: dict[str, list[tuple[int, float]]] = {mth: [] for mth in methods}
    for outer, (v0, v1) in OUTER_MONTHS.items():
        t0, t1 = INNER_TUNE[outer]
        vmask = (cmonth >= v0) & (cmonth <= v1)
        tmask = (cmonth >= t0) & (cmonth <= t1)
        base_norm, base_scale = normalize_prediction(cbase[vmask], "rms")
        yv = cy[vmask]
        base_score = cosine_uncentered(base_norm, yv)
        for mth in methods:
            # per-month drift estimate (rolling-origin)
            drift = np.zeros(vmask.sum())
            for m in range(v0, v1 + 1):
                est = mu_estimators(mu, m)[mth]
                drift[cmonth[vmask] == m] = est
            drift = drift - drift.mean()  # center
            # alpha selection on tune months
            tune_full = (cmonth >= t0) & (cmonth <= t1) & vmask
            tune_base, _ = normalize_prediction(cbase[tune_full], "rms")
            tune_drift = drift[tune_full[vmask]]
            tune_y = cy[tune_full]
            scores = [cosine_uncentered(tune_base + a * tune_drift, tune_y) for a in ALPHA_GRID]
            best_a = float(ALPHA_GRID[int(np.argmax(scores))])
            final = base_norm + best_a * drift
            score = cosine_uncentered(final, yv)
            delta = score - base_score
            # month-by-month delta
            for m in range(v0, v1 + 1):
                mm = cmonth[vmask] == m
                if mm.sum() > 100:
                    s0 = cosine_uncentered(base_norm[mm], yv[mm])
                    s1 = cosine_uncentered(final[mm], yv[mm])
                    month_deltas[mth].append((m, float(s1 - s0)))
            # activity split
            ahi = hi_act[vmask]
            d_hi = cosine_uncentered(final[ahi], yv[ahi]) - cosine_uncentered(base_norm[ahi], yv[ahi])
            d_lo = cosine_uncentered(final[~ahi], yv[~ahi]) - cosine_uncentered(base_norm[~ahi], yv[~ahi])
            summary_rows.append({
                "outer": outer, "method": mth, "alpha": best_a,
                "delta": float(delta), "score": float(score),
                "delta_hi": float(d_hi), "delta_lo": float(d_lo),
            })
            print(f"{outer} {mth:12s} alpha={best_a:.2f} delta={delta:+.6f} "
                  f"(hi={d_hi:+.6f} lo={d_lo:+.6f})")

    print("\n=== method summary (mean delta over outers) ===")
    for mth in methods:
        rows = [r for r in summary_rows if r["method"] == mth]
        mean_d = np.mean([r["delta"] for r in rows])
        pos = sum(1 for r in rows if r["delta"] > 0)
        md = month_deltas[mth]
        pos_months = sum(1 for _, d in md if d > 0)
        print(f"{mth:12s} mean_delta={mean_d:+.6f} outers_pos={pos}/4 months_pos={pos_months}/{len(md)}")

    import json
    (OUT / "results.json").write_text(json.dumps({
        "mu_mean": float(mu.mean()), "mu_std": float(mu.std()),
        "acf": acf, "sign_persistence": float(same), "trend_slope": float(b),
        "rows": summary_rows,
        "month_deltas": {k: [list(x) for x in v] for k, v in month_deltas.items()},
    }, indent=2), encoding="utf-8")
    print("\nwritten to", OUT)


if __name__ == "__main__":
    main()
