# -*- coding: utf-8 -*-
"""P3 shared helpers: feature artifact construction + generic frame loader.

Any new representation (SAE latent, state-concat, grid, sequence latent) is
packaged as an EventFlowFrame-style parquet + manifest and validated through
the same frozen residual protocol as M01-A (fit -> tune alpha -> refit -> outer).
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from mscapital.artifacts import array_hash, feature_hash
from mscapital.models.m01a import EventFlowFrame


def make_manifest(
    experiment_id: str,
    feature_names: tuple[str, ...],
    sample_id: np.ndarray,
    month: np.ndarray,
    target: np.ndarray,
    values: np.ndarray,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "experiment_id": experiment_id,
        "status": "complete",
        "feature_hash": feature_hash(list(feature_names)),
        "feature_names": list(feature_names),
        "diagnostics": {
            "rows": int(sample_id.size),
            "artifact_hashes": {
                "sample_id": array_hash(sample_id),
                "month": array_hash(month),
                "target": array_hash(target),
                "values": array_hash(values),
            },
            **(extra or {}),
        },
    }


def save_p3_features(
    path: str | Path,
    experiment_id: str,
    feature_names: tuple[str, ...],
    sample_id: np.ndarray,
    month: np.ndarray,
    target: np.ndarray,
    values: np.ndarray,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Write features as parquet + manifest in the M01-A artifact shape."""
    import polars as pl

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = {
        "sample_id": np.asarray(sample_id),
        "month": np.asarray(month),
        "target": np.asarray(target),
    }
    values = np.asarray(values, dtype=np.float32)
    for index, name in enumerate(feature_names):
        frame[name] = values[:, index]
    pl.DataFrame(frame).write_parquet(path)
    manifest = make_manifest(
        experiment_id, feature_names, sample_id, month, target, values, extra
    )
    (path.parent / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return manifest


def load_p3_frame(
    path: str | Path,
    feature_names: tuple[str, ...],
    *,
    require_experiment_id: str | None = None,
) -> EventFlowFrame:
    """Load a P3 feature artifact with the same strict validation as M01-A."""
    import polars as pl

    path = Path(path)
    manifest_path = path.parent / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"{path}: manifest.json is required")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if require_experiment_id is not None and manifest.get("experiment_id") != require_experiment_id:
        raise ValueError(
            f"{path}: expected experiment_id {require_experiment_id!r}, "
            f"got {manifest.get('experiment_id')!r}"
        )
    required = ("sample_id", "month", "target", *feature_names)
    frame = pl.read_parquet(path)
    missing = set(required) - set(frame.columns)
    if missing:
        raise ValueError(f"{path}: missing columns {sorted(missing)}")
    frame = frame.sort("sample_id")
    columns = {name: frame[name].to_numpy() for name in required}
    result = EventFlowFrame(
        columns["sample_id"],
        columns["month"],
        np.asarray(columns["target"], dtype=np.float64),
        np.column_stack([columns[name] for name in feature_names]).astype(np.float32),
        tuple(feature_names),
    )
    result.validate()
    expected = make_manifest(
        "x", feature_names, result.sample_id, result.month, result.target, result.values
    )
    diag = manifest.get("diagnostics", {})
    if diag.get("artifact_hashes") != expected["diagnostics"]["artifact_hashes"]:
        raise ValueError(f"{path}: artifact hashes do not match manifest")
    if diag.get("rows") != result.sample_id.size:
        raise ValueError(f"{path}: manifest row count mismatch")
    if manifest.get("feature_hash") != feature_hash(list(feature_names)):
        raise ValueError(f"{path}: feature hash mismatch")
    return result


def concat_frames(
    left: EventFlowFrame, right: EventFlowFrame
) -> EventFlowFrame:
    """Row-align two feature frames by sample_id (both must cover same ids)."""
    assert left.sample_id.size == right.sample_id.size
    order = np.argsort(left.sample_id, kind="mergesort")
    rorder = np.argsort(right.sample_id, kind="mergesort")
    assert np.array_equal(left.sample_id[order], right.sample_id[rorder])
    assert np.array_equal(left.month[order], right.month[rorder])
    assert np.array_equal(left.target[order], right.target[rorder])
    values = np.hstack([left.values[order], right.values[rorder]])
    names = tuple(left.feature_names) + tuple(right.feature_names)
    return EventFlowFrame(
        left.sample_id[order], left.month[order], left.target[order],
        values.astype(np.float32), names,
    )
