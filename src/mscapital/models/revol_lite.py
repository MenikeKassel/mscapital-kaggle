"""E01 ReVol-lite residual experiment under Protocol-v2."""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from ..artifacts import ExperimentManifest, array_hash, feature_hash
from ..diagnostics import drift_report, prediction_diagnostics
from ..features.revol_lite import revol_lite_feature_names
from ..metrics import cosine_uncentered
from ..residual import CanonicalOOF, outer_residual
from ..splits import NESTED_SPLITS
from .m01a import (
    ALPHA_GRID,
    M01AConfig,
    RESIDUAL_INNER_SPLITS,
    _apply_selected_scale,
    _load_outer_baseline,
    _take_features,
    fit_m01a_selection,
)


class RevolLiteFrame:
    """Array-backed E01 artifact sharing the M01 selection protocol seam."""

    def __init__(self, sample_id: np.ndarray, month: np.ndarray, target: np.ndarray,
                 values: np.ndarray, feature_names: tuple[str, ...]):
        self.sample_id = np.asarray(sample_id).reshape(-1)
        self.month = np.asarray(month).reshape(-1)
        self.target = np.asarray(target, dtype=np.float64).reshape(-1)
        self.values = np.asarray(values, dtype=np.float32)
        self.feature_names = tuple(feature_names)

    def validate(self) -> None:
        n = self.sample_id.size
        if self.month.size != n or self.target.size != n:
            raise ValueError("ReVol-lite identifiers, months and targets must align")
        if self.values.shape != (n, len(self.feature_names)):
            raise ValueError("ReVol-lite matrix shape does not match rows/features")
        if np.unique(self.sample_id).size != n:
            raise ValueError("ReVol-lite sample_id must be unique")
        if not np.isfinite(self.target).all() or not np.isfinite(self.values).all():
            raise ValueError("ReVol-lite targets and features must be finite")


def load_revol_lite_frame(path: str | Path) -> RevolLiteFrame:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("pyarrow is required to load ReVol-lite features") from exc
    path = Path(path)
    manifest_path = path.parent / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError("ReVol-lite manifest.json is required")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("experiment_id") != "e01-revol-lite-features" or manifest.get("status") != "complete":
        raise ValueError("ReVol-lite manifest identity/status is invalid")
    names = revol_lite_feature_names()
    required = ("sample_id", "month", "target", *names)
    table = pq.read_table(path, columns=list(required))
    if set(table.column_names) != set(required):
        raise ValueError("ReVol-lite artifact columns are incomplete")
    columns = {name: table[name].to_numpy(zero_copy_only=False) for name in required}
    order = np.argsort(columns["sample_id"], kind="stable")
    columns = {name: np.asarray(value)[order] for name, value in columns.items()}
    frame = RevolLiteFrame(
        columns["sample_id"], columns["month"], columns["target"],
        np.column_stack([columns[name] for name in names]), tuple(names),
    )
    frame.validate()
    if manifest.get("feature_hash") != feature_hash(names):
        raise ValueError("ReVol-lite feature hash is invalid")
    diagnostics = manifest.get("diagnostics", {})
    if diagnostics.get("rows") != frame.sample_id.size or diagnostics.get("feature_count") != len(names):
        raise ValueError("ReVol-lite manifest dimensions are invalid")
    expected = {
        "sample_id": array_hash(frame.sample_id), "month": array_hash(frame.month),
        "target": array_hash(frame.target.astype(np.float32)), "values": array_hash(frame.values),
    }
    if diagnostics.get("artifact_hashes") != expected:
        raise ValueError("ReVol-lite artifact hashes are invalid")
    return frame


def run_revol_lite_outer(
    canonical: CanonicalOOF,
    features: RevolLiteFrame,
    baseline_root: str | Path,
    output_root: str | Path,
    outer: str,
    *,
    config: M01AConfig = M01AConfig(),
) -> dict[str, Any]:
    """Run one E01 outer fold with exactly the frozen M01 residual protocol."""
    started = time.perf_counter()
    selection = fit_m01a_selection(canonical, features, outer, config=config)
    baseline = _load_outer_baseline(baseline_root, outer)
    x_outer, feature_month, feature_target = _take_features(features, baseline["sample_id"])
    if not np.array_equal(feature_month, baseline["month"]) or not np.array_equal(feature_target, baseline["target"]):
        raise ValueError(f"{outer}: ReVol-lite and frozen baseline labels must align")
    residual_prediction = np.asarray(selection.refit_model.predict(x_outer), dtype=np.float64)
    final = _apply_selected_scale(baseline["pred"], selection.baseline_scale) + selection.alpha * _apply_selected_scale(
        residual_prediction, selection.residual_scale
    )
    if not np.isfinite(final).all():
        raise ValueError(f"{outer}: ReVol-lite final predictions must be finite")
    baseline_score = cosine_uncentered(baseline["pred"], baseline["target"])
    final_score = cosine_uncentered(final, baseline["target"])
    diagnostics = prediction_diagnostics(final, baseline["target"], reference=baseline["pred"])
    diagnostics.update({
        "outer": outer, "beta": selection.beta, "alpha": selection.alpha,
        "best_iteration": selection.best_iteration,
        "baseline_scale": selection.baseline_scale, "residual_scale": selection.residual_scale,
        "inner_tune_score": selection.tune_score, "inner_tune_baseline_score": selection.tune_baseline_score,
        "baseline_score": baseline_score, "final_score": final_score,
        "delta_vs_baseline": final_score - baseline_score,
        "drift": drift_report(selection.tune_prediction, final), "rows": int(final.size),
        "method": "E01 ReVol-lite", "candidate_id": "e01-revol-lite",
    })
    output = Path(output_root) / "e01-revol-lite" / outer
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output / "inner_predictions.npz", sample_id=selection.tune_sample_id,
        month=selection.tune_month, target=selection.tune_target,
        baseline_oof=selection.tune_baseline_oof, residual_pred=selection.tune_residual_prediction,
        pred=selection.tune_prediction,
    )
    np.savez_compressed(
        output / "predictions.npz", sample_id=baseline["sample_id"], month=baseline["month"],
        target=baseline["target"], baseline_pred=baseline["pred"], residual_pred=residual_prediction,
        pred=final, split=np.full(final.size, f"{outer}:e01-revol-lite"),
    )
    config_payload = json.dumps(asdict(config), sort_keys=True, separators=(",", ":")).encode()
    manifest = ExperimentManifest(
        experiment_id=f"e01-revol-lite-{outer.lower()}", status="complete",
        config_hash=hashlib.sha256(config_payload).hexdigest(),
        data_fingerprints={
            "canonical_sample_id": array_hash(canonical.sample_id),
            "canonical_month": array_hash(canonical.month), "canonical_target": array_hash(canonical.target),
            "canonical_baseline_oof": array_hash(canonical.baseline_oof),
            "revol_lite_values": array_hash(features.values), "frozen_outer_baseline": array_hash(baseline["pred"]),
        }, feature_hash=feature_hash(features.feature_names),
        train_months=(21, int(np.max(outer_residual(canonical, outer)["month"]))),
        valid_months=NESTED_SPLITS[outer].outer_valid.as_tuple(), best_step=selection.best_iteration,
        scores={"cosine_uncentered": final_score}, diagnostics=diagnostics,
        runtime_seconds=time.perf_counter() - started,
    )
    manifest.write(output)
    (output / "report.md").write_text("\n".join([
        f"# E01 ReVol-lite - {outer}", "", f"- score: `{final_score:.9f}`",
        f"- delta vs frozen baseline: `{final_score - baseline_score:+.9f}`",
        f"- beta / alpha: `{selection.beta:.9g}` / `{selection.alpha:.2f}`",
        f"- best iteration: `{selection.best_iteration}`", "",
    ]), encoding="utf-8")
    return diagnostics | {"output": str(output)}


def summarize_revol_lite(artifact_root: str | Path) -> dict[str, Any]:
    root = Path(artifact_root) / "e01-revol-lite"
    rows: list[dict[str, Any]] = []
    for outer in ("PSEUDO", "H2", "T3", "T4"):
        directory = root / outer
        manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
        if manifest.get("status") != "complete":
            raise ValueError(f"{outer}: E01 artifact is incomplete")
        with np.load(directory / "predictions.npz") as source:
            required = {"sample_id", "month", "target", "baseline_pred", "residual_pred", "pred", "split"}
            if set(source.files) != required:
                raise ValueError(f"{outer}: E01 prediction schema is invalid")
            artifact = {key: np.asarray(source[key]) for key in required}
        if len({value.size for value in artifact.values()}) != 1:
            raise ValueError(f"{outer}: E01 prediction lengths differ")
        if not all(np.isfinite(artifact[key]).all() for key in ("target", "baseline_pred", "residual_pred", "pred")):
            raise ValueError(f"{outer}: E01 predictions contain non-finite values")
        expected_months = set(range(NESTED_SPLITS[outer].outer_valid.start, NESTED_SPLITS[outer].outer_valid.end + 1))
        if set(artifact["month"].tolist()) != expected_months:
            raise ValueError(f"{outer}: E01 prediction months are invalid")
        if np.unique(artifact["sample_id"]).size != artifact["sample_id"].size:
            raise ValueError(f"{outer}: E01 prediction sample_id values are not unique")
        if not np.array_equal(artifact["split"], np.full(artifact["pred"].size, f"{outer}:e01-revol-lite")):
            raise ValueError(f"{outer}: E01 prediction split labels are invalid")
        if manifest.get("experiment_id") != f"e01-revol-lite-{outer.lower()}":
            raise ValueError(f"{outer}: E01 manifest identity is invalid")
        diagnostics = dict(manifest.get("diagnostics", {}))
        baseline_score = cosine_uncentered(artifact["baseline_pred"], artifact["target"])
        final_score = cosine_uncentered(artifact["pred"], artifact["target"])
        if not np.isclose(final_score, diagnostics.get("final_score"), atol=1e-12, rtol=0.0):
            raise ValueError(f"{outer}: E01 score does not match manifest")
        rows.append({**diagnostics, "baseline_score": baseline_score, "final_score": final_score,
                     "delta_vs_baseline": final_score - baseline_score})
    deltas = np.asarray([row["delta_vs_baseline"] for row in rows], dtype=float)
    drift_ok = all(
        0.67 <= row.get("drift", {}).get("std_test_over_valid", 0.0) <= 1.50
        and 0.50 <= row.get("drift", {}).get("abs_p99_test_over_valid", 0.0) <= 2.00
        for row in rows
    )
    gate = {
        "pseudo_delta_at_least_0_0015": bool(deltas[0] >= 0.0015),
        "positive_outers": int((deltas > 0).sum()), "worst_delta": float(deltas.min()),
        "drift_ok": bool(drift_ok), "finite_ok": True,
    }
    gate["passed"] = bool(gate["pseudo_delta_at_least_0_0015"] and gate["positive_outers"] >= 3 and gate["worst_delta"] >= -0.0005 and gate["drift_ok"])
    return {"method": "E01 ReVol-lite", "rows": rows, "mean_delta": float(deltas.mean()), "gate": gate}
