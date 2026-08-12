"""Prediction and feature distribution diagnostics."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from .metrics import cosine_uncentered


def distribution_report(values: object) -> dict[str, Any]:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    if arr.size == 0:
        raise ValueError("cannot report an empty distribution")
    finite = np.isfinite(arr)
    clean = arr[finite]
    if clean.size == 0:
        return {"count": int(arr.size), "finite": 0, "nan_or_inf": int(arr.size)}
    q = np.quantile(clean, [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99])
    return {
        "count": int(arr.size),
        "finite": int(clean.size),
        "nan_or_inf": int((~finite).sum()),
        "mean": float(clean.mean()),
        "std": float(clean.std()),
        "min": float(clean.min()),
        "max": float(clean.max()),
        "q01": float(q[0]),
        "q05": float(q[1]),
        "q25": float(q[2]),
        "q50": float(q[3]),
        "q75": float(q[4]),
        "q95": float(q[5]),
        "q99": float(q[6]),
    }


def prediction_diagnostics(
    pred: object,
    target: object | None = None,
    *,
    reference: object | None = None,
) -> dict[str, Any]:
    report: dict[str, Any] = {"prediction": distribution_report(pred)}
    if target is not None:
        report["target"] = distribution_report(target)
        report["cosine_uncentered"] = cosine_uncentered(pred, target)
    if reference is not None:
        p, r = np.asarray(pred, dtype=float), np.asarray(reference, dtype=float)
        if p.shape != r.shape:
            raise ValueError("reference shape must match prediction")
        p, r = p.reshape(-1), r.reshape(-1)
        if p.size == 0 or not np.isfinite(p).all() or not np.isfinite(r).all():
            report["corr_reference"] = None
        elif np.std(p) == 0.0 or np.std(r) == 0.0:
            report["corr_reference"] = 0.0
        else:
            report["corr_reference"] = float(np.corrcoef(p, r)[0, 1])
    return report


def drift_report(valid: object, test: object) -> dict[str, float]:
    v, t = distribution_report(valid), distribution_report(test)
    v_std = float(v.get("std", 0.0))
    t_std = float(t.get("std", 0.0))
    v_p99 = max(abs(float(v.get("q99", 0.0))), abs(float(v.get("q01", 0.0))))
    t_p99 = max(abs(float(t.get("q99", 0.0))), abs(float(t.get("q01", 0.0))))
    return {
        "std_test_over_valid": 0.0 if v_std == 0 else t_std / v_std,
        "abs_p99_test_over_valid": 0.0 if v_p99 == 0 else t_p99 / v_p99,
        "mean_test_minus_valid": float(t.get("mean", 0.0) - v.get("mean", 0.0)),
    }
