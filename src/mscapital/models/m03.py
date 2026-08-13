"""M03 depth-2 Path Signature residual experiment."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from ..artifacts import array_hash, feature_hash
from ..features.path_signature import path_signature_feature_names
from ..residual import CanonicalOOF
from .m01a import M01AConfig
from .m02 import GeometryFrame, run_m02_outer, summarize_m02


def load_path_signature_frame(path: str | Path) -> GeometryFrame:
    """Load and integrity-check the fixed 112-column M03 artifact."""

    try:
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("M03 requires pyarrow") from exc
    path = Path(path)
    manifest_path = path.parent / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError("M03 path-signature manifest.json is required")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("experiment_id") != "m03-path-signature-features"
        or manifest.get("status") != "complete"
    ):
        raise ValueError("M03 path-signature manifest identity/status is invalid")
    names = path_signature_feature_names()
    required = ("sample_id", "month", "target", *names)
    table = pq.read_table(path, columns=list(required))
    if set(table.column_names) != set(required):
        raise ValueError("M03 path-signature feature columns are incomplete")
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
        raise ValueError("M03 path-signature manifest row count is invalid")
    if manifest.get("feature_hash") != feature_hash(list(names)):
        raise ValueError("M03 path-signature feature hash is invalid")
    expected = {
        "sample_id": array_hash(frame.sample_id), "month": array_hash(frame.month),
        "target": array_hash(columns["target"]), "values": array_hash(frame.values),
    }
    if diagnostics.get("artifact_hashes") != expected:
        raise ValueError("M03 path-signature artifact hashes are invalid")
    return frame


def run_m03_outer(
    canonical: CanonicalOOF,
    features: GeometryFrame,
    baseline_root: str | Path,
    output_root: str | Path,
    outer: str,
    *,
    config: M01AConfig = M01AConfig(),
) -> dict[str, Any]:
    """Run one M03 fold through the frozen M01-A residual protocol."""

    return run_m02_outer(
        canonical, features, baseline_root, output_root, outer, config=config,
        method_id="m03-path-signature", output_subdir="m03-path-signature",
        split_label="m03-path-signature", report_label="M03 depth-2 Path Signature",
    )


def summarize_m03(artifact_root: str | Path) -> dict[str, Any]:
    """Replay and summarize the four fixed M03 outer-fold artifacts."""

    return summarize_m02(
        artifact_root, output_subdir="m03-path-signature",
        split_label="m03-path-signature", method="M03 depth-2 Path Signature",
    )
