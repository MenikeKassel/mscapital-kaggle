# -*- coding: utf-8 -*-
"""Recompute P6R-00 V2-capped / beta1 variants for all outers and merge metrics.

The main runner skips variants on the resume path; this script fills them in
without repeating the 8-candidate KNN sweep. Diagnostics only (not gated).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[0]))

from p6r_00_retrieval_residual import (
    INNER_TRAIN,
    OUTER_EVAL,
    TUNE,
    VISIBLE_END,
    StateKNN,
    align_ids,
    bank_mask,
    blend_with_scales,
    fit_state_scaler,
    load_c4_baseline,
    query_months,
    retrieve_residual_mean,
    select_alpha,
    standardize,
    take_context,
)
from mscapital.metrics import cosine_uncentered
from mscapital.residual import load_canonical_oof_artifact, outer_residual
from mscapital.models.revol_lite import load_revol_lite_frame

CANONICAL = Path("output/canonical_residual_oof/canonical_residual_oof.npz")
FEATURES = Path("output/e01_revol_lite_features/revol_lite_train.parquet")
C4_ROOT = Path("output/c4_protocol_closed_final/clean-baseline-v2")
OUT_ROOT = Path("output/p6r_00")


def main() -> None:
    canonical = load_canonical_oof_artifact(CANONICAL)
    frame = load_revol_lite_frame(FEATURES)
    z_all = take_context(frame, canonical.sample_id)
    months = np.asarray(canonical.month)
    target = np.asarray(canonical.target, dtype=np.float64)
    baseline = np.asarray(canonical.baseline_oof, dtype=np.float64)
    metrics_path = OUT_ROOT / "metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))

    for outer in ("PSEUDO", "H2", "T3", "T4"):
        view = outer_residual(canonical, outer)
        beta = float(view["beta"])
        r_resid = target - beta * baseline
        it_lo, it_hi = INNER_TRAIN[outer]
        it_mask = (months >= it_lo) & (months <= it_hi)
        mu, sd = fit_state_scaler(z_all[it_mask])
        z_std = standardize(z_all, mu, sd)
        c4 = load_c4_baseline(C4_ROOT, outer)
        c4_idx = align_ids(c4["sample_id"], canonical.sample_id)
        base_c4 = cosine_uncentered(c4["pred"], c4["target"])
        tune_lo, tune_hi = TUNE[outer]
        tune_mask = (months >= tune_lo) & (months <= tune_hi)
        q_months = query_months(months, tune_lo)

        results = metrics[outer]
        config_keys = [k for k in results if not k.startswith(("v2_", "beta1"))]
        best_key = max(config_keys, key=lambda k: results[k]["tune_score"])
        best = results[best_key]
        k, metric = best["k"], best["metric"]
        cap = VISIBLE_END[outer]

        # V2: capped bank
        r_hat_v2 = np.full(months.size, np.nan)
        for m in q_months:
            qmask = months == m
            bmask = bank_mask(months, m, cap_month=cap)
            if not bmask.any():
                continue
            knn = StateKNN(metric).fit(z_std[bmask], k)
            idx, dists, w = knn.query(z_std[qmask], k)
            r_hat_v2[qmask] = retrieve_residual_mean(idx, w, r_resid)
        sel_v2 = select_alpha(baseline[tune_mask], r_hat_v2[tune_mask], target[tune_mask], np.array([0.05, 0.10, 0.15, 0.20, 0.30]))
        y_hat_v2 = blend_with_scales(c4["pred"], r_hat_v2[c4_idx], sel_v2["baseline_rms"], sel_v2["residual_rms"], sel_v2["alpha"])
        results["v2_capped"] = {
            "variant": f"capped_{cap}_bank_of_{best_key}",
            "delta_c4": cosine_uncentered(y_hat_v2, c4["target"]) - base_c4,
            "alpha": sel_v2["alpha"],
        }

        # beta=1 residual variant
        r1 = target - baseline
        r_hat_1 = np.full(months.size, np.nan)
        for m in q_months:
            qmask = months == m
            bmask = bank_mask(months, m)
            knn = StateKNN(metric).fit(z_std[bmask], k)
            idx, dists, w = knn.query(z_std[qmask], k)
            r_hat_1[qmask] = retrieve_residual_mean(idx, w, r1)
        sel_1 = select_alpha(baseline[tune_mask], r_hat_1[tune_mask], target[tune_mask], np.array([0.05, 0.10, 0.15, 0.20, 0.30]))
        y_hat_1 = blend_with_scales(c4["pred"], r_hat_1[c4_idx], sel_1["baseline_rms"], sel_1["residual_rms"], sel_1["alpha"])
        results["beta1_variant"] = {
            "variant": f"residual_y_minus_y0_of_{best_key}",
            "delta_c4": cosine_uncentered(y_hat_1, c4["target"]) - base_c4,
            "alpha": sel_1["alpha"],
        }
        print(f"{outer}: best={best_key} v2_capped={results['v2_capped']['delta_c4']:+.6f} (a={results['v2_capped']['alpha']}) "
              f"beta1={results['beta1_variant']['delta_c4']:+.6f} (a={results['beta1_variant']['alpha']})", flush=True)

    metrics_path.write_text(json.dumps(metrics, indent=2, default=float), encoding="utf-8")
    print("merged ->", metrics_path)


if __name__ == "__main__":
    main()
