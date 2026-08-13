"""M02 Market-Centered LOB Geometry residual evaluation.

M02 deliberately reuses the frozen M01-A CatBoost/alpha protocol.  Only the
representation changes; split registration, beta estimation, RMS direction
normalisation and outer diagnostics remain identical.
"""

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
from ..features.lob_geometry import geometry_feature_names
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


class GeometryFrame:
    """Array-backed Geometry artifact with the same protocol seam as EventFlow."""

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
            raise ValueError("Geometry identifiers, months and targets must align")
        if self.values.shape != (n, len(self.feature_names)):
            raise ValueError("Geometry matrix shape does not match rows/features")
        if np.unique(self.sample_id).size != n:
            raise ValueError("Geometry sample_id must be unique")
        if not np.isfinite(self.target).all() or not np.isfinite(self.values).all():
            raise ValueError("Geometry targets and features must be finite")


def load_geometry_frame(path: str | Path) -> GeometryFrame:
    """Load and verify the streaming M02 Parquet artifact without Polars."""
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("M02 requires pyarrow to load Geometry features") from exc
    path = Path(path)
    manifest_path = path.parent / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError("M02 Geometry manifest.json is required")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("experiment_id") != "m02-geometry-features" or manifest.get("status") != "complete":
        raise ValueError("M02 Geometry manifest identity/status is invalid")
    names = geometry_feature_names()
    required = ("sample_id", "month", "target", *names)
    table = pq.read_table(path, columns=list(required))
    if set(table.column_names) != set(required):
        raise ValueError("M02 Geometry artifact columns are incomplete")
    columns = {name: table[name].to_numpy(zero_copy_only=False) for name in required}
    order = np.argsort(columns["sample_id"], kind="mergesort")
    columns = {name: np.asarray(value)[order] for name, value in columns.items()}
    frame = GeometryFrame(
        columns["sample_id"], columns["month"], columns["target"],
        np.column_stack([columns[name] for name in names]), tuple(names),
    )
    frame.validate()
    if manifest.get("feature_hash") != feature_hash(list(names)):
        raise ValueError("M02 Geometry feature hash is invalid")
    diagnostics = manifest.get("diagnostics", {})
    if diagnostics.get("rows") != frame.sample_id.size:
        raise ValueError("M02 Geometry manifest row count is invalid")
    expected = {
        "sample_id": array_hash(frame.sample_id), "month": array_hash(frame.month),
        "target": array_hash(columns["target"]), "values": array_hash(frame.values),
    }
    if diagnostics.get("artifact_hashes") != expected:
        raise ValueError("M02 Geometry artifact hashes are invalid")
    return frame


def run_m02_outer(
    canonical: CanonicalOOF,
    features: GeometryFrame,
    baseline_root: str | Path,
    output_root: str | Path,
    outer: str,
    *,
    config: M01AConfig = M01AConfig(),
    method_id: str = "m02-geometry",
    output_subdir: str = "m02-geometry",
    split_label: str = "m02-geometry",
    report_label: str = "M02 Geometry",
) -> dict[str, Any]:
    """Run one M02 outer fold; selection sees only historical canonical OOF."""
    started = time.perf_counter()
    selection = fit_m01a_selection(canonical, features, outer, config=config)
    baseline = _load_outer_baseline(baseline_root, outer)
    x_outer, feature_month, feature_target = _take_features(features, baseline["sample_id"])
    if not np.array_equal(feature_month, baseline["month"]) or not np.array_equal(feature_target, baseline["target"]):
        raise ValueError(f"{outer}: Geometry and frozen baseline labels must align")
    residual_prediction = np.asarray(selection.refit_model.predict(x_outer), dtype=np.float64)
    final = _apply_selected_scale(baseline["pred"], selection.baseline_scale) + selection.alpha * _apply_selected_scale(
        residual_prediction, selection.residual_scale
    )
    if not np.isfinite(final).all():
        raise ValueError(f"{outer}: final M02 predictions must be finite")
    baseline_score = cosine_uncentered(baseline["pred"], baseline["target"])
    final_score = cosine_uncentered(final, baseline["target"])
    diagnostics = prediction_diagnostics(final, baseline["target"], reference=baseline["pred"])
    diagnostics.update({
        "outer": outer, "beta": selection.beta, "alpha": selection.alpha,
        "best_iteration": selection.best_iteration,
        "baseline_scale": selection.baseline_scale, "residual_scale": selection.residual_scale,
        "inner_tune_score": selection.tune_score,
        "inner_tune_baseline_score": selection.tune_baseline_score,
        "baseline_score": baseline_score, "final_score": final_score,
        "delta_vs_baseline": final_score - baseline_score,
        "drift": drift_report(selection.tune_prediction, final), "rows": int(final.size),
        "lb142_prediction_corr": None,
        "lb142_status": "no_outer_aligned_reference_provided",
    })
    output = Path(output_root) / output_subdir / outer
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output / "inner_predictions.npz", sample_id=selection.tune_sample_id,
        month=selection.tune_month, target=selection.tune_target,
        baseline_oof=selection.tune_baseline_oof,
        residual_pred=selection.tune_residual_prediction, pred=selection.tune_prediction,
    )
    np.savez_compressed(
        output / "predictions.npz", sample_id=baseline["sample_id"], month=baseline["month"],
        target=baseline["target"], baseline_pred=baseline["pred"],
        residual_pred=residual_prediction, pred=final,
        split=np.full(final.size, f"{outer}:{split_label}"),
    )
    (output / "training_history.json").write_text(json.dumps({
        "outer": outer, "beta": selection.beta, "best_iteration": selection.best_iteration,
        "alpha": selection.alpha, "alpha_grid": ALPHA_GRID.tolist(),
        "baseline_scale": selection.baseline_scale, "residual_scale": selection.residual_scale,
        "inner_tune_score": selection.tune_score,
    }, indent=2), encoding="utf-8")
    config_payload = json.dumps(asdict(config), sort_keys=True, separators=(",", ":")).encode()
    view = outer_residual(canonical, outer)
    manifest = ExperimentManifest(
        experiment_id=f"{method_id}-{outer.lower()}", status="complete",
        config_hash=hashlib.sha256(config_payload).hexdigest(),
        data_fingerprints={
            "canonical_sample_id": array_hash(canonical.sample_id),
            "canonical_month": array_hash(canonical.month),
            "canonical_target": array_hash(canonical.target),
            "canonical_baseline_oof": array_hash(canonical.baseline_oof),
            "geometry_values": array_hash(features.values),
            "frozen_outer_baseline": array_hash(baseline["pred"]),
        },
        feature_hash=feature_hash(list(features.feature_names)),
        train_months=(21, int(np.max(view["month"]))),
        valid_months=NESTED_SPLITS[outer].outer_valid.as_tuple(), best_step=selection.best_iteration,
        scores={"cosine_uncentered": final_score}, diagnostics=diagnostics,
        runtime_seconds=time.perf_counter() - started,
    )
    manifest.write(output)
    (output / "report.md").write_text("\n".join([
        f"# {report_label} - {outer}", "",
        f"- score: `{final_score:.9f}`",
        f"- delta vs frozen baseline: `{final_score - baseline_score:+.9f}`",
        f"- beta / alpha: `{selection.beta:.9g}` / `{selection.alpha:.2f}`",
        f"- best iteration: `{selection.best_iteration}`", "",
    ]), encoding="utf-8")
    return diagnostics | {"output": str(output)}


def summarize_m02(
    artifact_root: str | Path,
    *,
    output_subdir: str = "m02-geometry",
    split_label: str = "m02-geometry",
    method: str = "M02 Geometry",
) -> dict[str, Any]:
    """Validate and summarize all four M02 outer artifacts."""
    root = Path(artifact_root) / output_subdir
    rows: list[dict[str, Any]] = []
    for outer in ("PSEUDO", "H2", "T3", "T4"):
        directory = root / outer
        manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
        if manifest.get("status") != "complete":
            raise ValueError(f"{outer}: M02 artifact is incomplete")
        diagnostics = dict(manifest.get("diagnostics", {}))
        with np.load(directory / "predictions.npz") as source:
            required = {"sample_id", "month", "target", "baseline_pred", "residual_pred", "pred", "split"}
            if set(source.files) != required:
                raise ValueError(f"{outer}: M02 predictions schema is invalid")
            artifact = {key: np.asarray(source[key]) for key in required}
        if len({value.size for value in artifact.values()}) != 1:
            raise ValueError(f"{outer}: M02 prediction lengths differ")
        if np.unique(artifact["sample_id"]).size != artifact["sample_id"].size:
            raise ValueError(f"{outer}: M02 sample_id values are not unique")
        if not all(np.isfinite(artifact[key]).all() for key in ("target", "baseline_pred", "residual_pred", "pred")):
            raise ValueError(f"{outer}: M02 predictions contain non-finite values")
        expected_months = set(range(NESTED_SPLITS[outer].outer_valid.start, NESTED_SPLITS[outer].outer_valid.end + 1))
        if set(artifact["month"].tolist()) != expected_months:
            raise ValueError(f"{outer}: M02 prediction months are invalid")
        if not np.array_equal(artifact["split"], np.full(artifact["pred"].size, f"{outer}:{split_label}")):
            raise ValueError(f"{outer}: M02 prediction split labels are invalid")
        baseline_score = cosine_uncentered(artifact["baseline_pred"], artifact["target"])
        final_score = cosine_uncentered(artifact["pred"], artifact["target"])
        if not np.isclose(baseline_score, diagnostics.get("baseline_score"), atol=1e-12, rtol=0.0):
            raise ValueError(f"{outer}: M02 baseline score does not match manifest")
        if not np.isclose(final_score, diagnostics.get("final_score"), atol=1e-12, rtol=0.0):
            raise ValueError(f"{outer}: M02 final score does not match manifest")
        if not np.isclose(final_score - baseline_score, diagnostics.get("delta_vs_baseline"), atol=1e-12, rtol=0.0):
            raise ValueError(f"{outer}: M02 delta does not match manifest")
        with np.load(directory / "inner_predictions.npz") as source:
            inner = {key: np.asarray(source[key]) for key in source.files}
        if not np.isclose(
            np.corrcoef(artifact["pred"], artifact["baseline_pred"])[0, 1],
            diagnostics.get("corr_reference"), atol=1e-12, rtol=0.0,
        ):
            raise ValueError(f"{outer}: M02 reference correlation does not match manifest")
        replay_drift = drift_report(inner["pred"], artifact["pred"])
        for key, value in replay_drift.items():
            if not np.isclose(value, diagnostics.get("drift", {}).get(key), atol=1e-12, rtol=0.0):
                raise ValueError(f"{outer}: M02 drift does not match manifest")
        diagnostics["finite_ok"] = True
        rows.append(diagnostics)
    deltas = np.asarray([row["delta_vs_baseline"] for row in rows], dtype=float)
    drift_ok = all(0.67 <= row["drift"]["std_test_over_valid"] <= 1.50 and 0.50 <= row["drift"]["abs_p99_test_over_valid"] <= 2.00 for row in rows)
    gate = {
        "pseudo_delta_at_least_0_0015": bool(rows[0]["delta_vs_baseline"] >= 0.0015),
        "positive_outers": int((deltas > 0).sum()), "worst_delta": float(deltas.min()),
        "drift_ok": bool(drift_ok), "finite_ok": True,
    }
    gate["passed"] = bool(gate["pseudo_delta_at_least_0_0015"] and gate["positive_outers"] >= 3 and gate["worst_delta"] >= -0.0005 and drift_ok)
    return {"method": method, "rows": rows, "mean_delta": float(deltas.mean()), "gate": gate}
