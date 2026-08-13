"""M02-T temporal dependency residual experiment."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from ..artifacts import array_hash, feature_hash
from ..features.geometry_temporal import geometry_temporal_feature_names
from ..metrics import cosine_uncentered
from ..residual import CanonicalOOF
from .m01a import M01AConfig
from .m02 import GeometryFrame, _load_outer_baseline, run_m02_outer, summarize_m02


def load_geometry_temporal_frame(path: str | Path) -> GeometryFrame:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("M02-T requires pyarrow") from exc
    path = Path(path)
    manifest_path = path.parent / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError("M02-T manifest.json is required")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("experiment_id") != "m02-t-geometry-features" or manifest.get("status") != "complete":
        raise ValueError("M02-T manifest identity/status is invalid")
    names = geometry_temporal_feature_names()
    required = ("sample_id", "month", "target", *names)
    table = pq.read_table(path, columns=list(required))
    if set(table.column_names) != set(required):
        raise ValueError("M02-T feature columns are incomplete")
    columns = {name: table[name].to_numpy(zero_copy_only=False) for name in required}
    order = np.argsort(columns["sample_id"], kind="mergesort")
    columns = {name: np.asarray(value)[order] for name, value in columns.items()}
    frame = GeometryFrame(
        columns["sample_id"], columns["month"], columns["target"],
        np.column_stack([columns[name] for name in names]), tuple(names),
    )
    frame.validate()
    diagnostics = manifest.get("diagnostics", {})
    if diagnostics.get("rows") != frame.sample_id.size:
        raise ValueError("M02-T manifest row count is invalid")
    if manifest.get("feature_hash") != feature_hash(list(names)):
        raise ValueError("M02-T feature hash is invalid")
    expected = {
        "sample_id": array_hash(frame.sample_id), "month": array_hash(frame.month),
        "target": array_hash(columns["target"]), "values": array_hash(frame.values),
    }
    if diagnostics.get("artifact_hashes") != expected:
        raise ValueError("M02-T feature artifact hashes are invalid")
    return frame


def run_m02t_outer(
    canonical: CanonicalOOF,
    features: GeometryFrame,
    baseline_root: str | Path,
    m02_base_root: str | Path,
    output_root: str | Path,
    outer: str,
    *,
    config: M01AConfig = M01AConfig(),
) -> dict[str, Any]:
    result = run_m02_outer(
        canonical, features, baseline_root, output_root, outer, config=config,
        method_id="m02-t", output_subdir="m02-t", split_label="m02-t",
        report_label="M02-T temporal Geometry",
    )
    output = Path(output_root) / "m02-t" / outer
    base_path = Path(m02_base_root) / "m02-geometry" / outer / "predictions.npz"
    if not base_path.exists():
        raise FileNotFoundError(f"{outer}: M02-base prediction artifact is required for attribution")
    with np.load(output / "predictions.npz") as temporal_source, np.load(base_path) as base_source:
        temporal = {key: np.asarray(temporal_source[key]) for key in temporal_source.files}
        base = {key: np.asarray(base_source[key]) for key in base_source.files}
    if not np.array_equal(temporal["sample_id"], base["sample_id"]) or not np.array_equal(temporal["month"], base["month"]):
        raise ValueError(f"{outer}: M02-T and M02-base IDs/months are not aligned")
    base_score = cosine_uncentered(base["pred"], base["target"])
    temporal_score = cosine_uncentered(temporal["pred"], temporal["target"])
    base_corr = float(np.corrcoef(temporal["pred"], base["pred"])[0, 1])
    diagnostics = dict(result)
    diagnostics.update({
        "m02_base_score": base_score,
        "delta_vs_m02_base": temporal_score - base_score,
        "m02_base_prediction_corr": base_corr,
    })
    manifest_path = output / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.setdefault("diagnostics", {}).update({
        "m02_base_score": base_score,
        "delta_vs_m02_base": temporal_score - base_score,
        "m02_base_prediction_corr": base_corr,
    })
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    (output / "report.md").write_text("\n".join([
        f"# M02-T temporal Geometry - {outer}", "",
        f"- final score: `{temporal_score:.9f}`",
        f"- delta vs frozen baseline: `{result['delta_vs_baseline']:+.9f}`",
        f"- delta vs M02-base: `{temporal_score - base_score:+.9f}`",
        f"- M02-T / M02-base prediction correlation: `{base_corr:.6f}`", "",
    ]), encoding="utf-8")
    return diagnostics


def summarize_m02t(artifact_root: str | Path) -> dict[str, Any]:
    result = summarize_m02(
        artifact_root, output_subdir="m02-t", split_label="m02-t", method="M02-T temporal Geometry"
    )
    rows = result["rows"]
    result["mean_delta_vs_m02_base"] = float(np.mean([row["delta_vs_m02_base"] for row in rows]))
    result["m02_base_prediction_corr"] = [row["m02_base_prediction_corr"] for row in rows]
    return result
