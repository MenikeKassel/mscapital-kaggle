"""Fixed temporal dependency features for the M02-T experiment.

The implementation intentionally keeps the representation small and frozen:
an as-of one-second quote path over the final 60 seconds, followed by four
fixed trailing windows.  No target, outer-fold state, or feature selection is
used while building the artifact.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Mapping

import numpy as np

from ..artifacts import ExperimentManifest, array_hash, feature_hash
from .lob_geometry import geometry_feature_names


WINDOWS = (5, 15, 30, 60)
BASE_FEATURE_COUNT = 21
TEMPORAL_FEATURES_PER_WINDOW = 16
TEMPORAL_FEATURE_COUNT = len(WINDOWS) * TEMPORAL_FEATURES_PER_WINDOW

_MARKET_COLUMNS = (
    "sample_id", "seconds_before_predict", "bid_price_1", "bid_volume_1",
    "ask_price_1", "ask_volume_1", "bid_price_2", "bid_volume_2",
    "ask_price_2", "ask_volume_2",
)


def temporal_feature_names() -> tuple[str, ...]:
    names: list[str] = []
    for window in WINDOWS:
        prefix = f"geom_t{window}"
        names.extend((f"{prefix}_coverage", f"{prefix}_log_updates"))
        names.extend(
            f"{prefix}_corr_{pair}"
            for pair in ("bid_levels", "ask_levels", "level_imbalance", "side_level")
        )
        names.extend(
            f"{prefix}_leadlag_{pair}"
            for pair in ("bid_levels", "ask_levels", "level_imbalance", "side_level")
        )
        names.extend(
            f"{prefix}_{signal}_{stat}"
            for signal in ("side_imbalance", "level_imbalance")
            for stat in ("mean", "std", "last_first")
        )
    return tuple(names)


def geometry_temporal_feature_names() -> tuple[str, ...]:
    return tuple(geometry_feature_names()) + temporal_feature_names()


def _corr(left: np.ndarray, right: np.ndarray) -> float:
    mask = np.isfinite(left) & np.isfinite(right)
    if int(mask.sum()) < 3:
        return 0.0
    x, y = left[mask].astype(float), right[mask].astype(float)
    x -= x.mean()
    y -= y.mean()
    denominator = float(np.sqrt(np.dot(x, x) * np.dot(y, y)))
    if denominator <= 1e-8:
        return 0.0
    return float(np.dot(x, y) / denominator)


def _lead_lag(left: np.ndarray, right: np.ndarray) -> float:
    if left.size < 4 or right.size < 4:
        return 0.0
    return _corr(left[:-1], right[1:]) - _corr(left[1:], right[:-1])


def _valid_quote(row_values: Mapping[str, np.ndarray]) -> np.ndarray:
    bid1, ask1 = row_values["bid_price_1"], row_values["ask_price_1"]
    bid2, ask2 = row_values["bid_price_2"], row_values["ask_price_2"]
    volumes = (
        row_values["bid_volume_1"], row_values["ask_volume_1"],
        row_values["bid_volume_2"], row_values["ask_volume_2"],
    )
    valid = np.isfinite(bid1) & np.isfinite(ask1) & np.isfinite(bid2) & np.isfinite(ask2)
    valid &= np.isfinite(row_values["seconds_before_predict"])
    valid &= row_values["seconds_before_predict"] >= 0.0
    valid &= bid2 <= bid1
    valid &= bid1 < ask1
    valid &= ask1 <= ask2
    for value in volumes:
        valid &= np.isfinite(value) & (value >= 0.0)
    return valid


def _quote_channels(row_values: Mapping[str, np.ndarray]) -> np.ndarray:
    bv1 = row_values["bid_volume_1"].astype(float)
    av1 = row_values["ask_volume_1"].astype(float)
    bv2 = row_values["bid_volume_2"].astype(float)
    av2 = row_values["ask_volume_2"].astype(float)
    l1 = np.maximum(bv1 + av1, 1e-12)
    l2 = np.maximum(bv2 + av2, 1e-12)
    total = np.maximum(l1 + l2, 1e-12)
    # bid-L1 share, ask-L1 share, bid-L2 share, ask-L2 share,
    # L1 imbalance, L2 imbalance, side imbalance, level imbalance.
    return np.column_stack((
        bv1 / total, av1 / total, bv2 / total, av2 / total,
        (bv1 - av1) / l1, (bv2 - av2) / l2,
        (bv1 + bv2 - av1 - av2) / total, (l1 - l2) / total,
    ))


def _asof_path(row_values: Mapping[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray, int]:
    seconds = np.asarray(row_values["seconds_before_predict"], dtype=float)
    valid = _valid_quote(row_values) & (seconds <= 60.0)
    if not valid.any():
        return np.zeros((61, 8), dtype=np.float64), np.zeros(61, dtype=bool), 0
    filtered = {name: np.asarray(value)[valid] for name, value in row_values.items()}
    seconds = filtered["seconds_before_predict"].astype(float)
    order = np.argsort(seconds, kind="mergesort")
    seconds = seconds[order]
    channels = _quote_channels({name: np.asarray(value)[order] for name, value in filtered.items()})
    grid = np.arange(60.0, -1.0, -1.0)
    path = np.zeros((grid.size, channels.shape[1]), dtype=np.float64)
    covered = np.zeros(grid.size, dtype=bool)
    positions = np.searchsorted(seconds, grid, side="left")
    valid_positions = positions < seconds.size
    path[valid_positions] = channels[positions[valid_positions]]
    covered[valid_positions] = True
    return path, covered, int(seconds.size)


def _window_features(path: np.ndarray, covered: np.ndarray, update_count: int, window: int) -> np.ndarray:
    start = 60 - window
    segment = path[start:]
    segment_covered = covered[start:]
    output = np.zeros(TEMPORAL_FEATURES_PER_WINDOW, dtype=np.float32)
    output[0] = float(segment_covered.mean())
    output[1] = float(np.log1p(update_count))
    pairs = ((0, 2), (1, 3), (4, 5), (6, 7))
    offset = 2
    for left, right in pairs:
        output[offset] = _corr(segment[:, left], segment[:, right])
        offset += 1
    for left, right in pairs:
        output[offset] = _lead_lag(segment[:, left], segment[:, right])
        offset += 1
    for signal in (6, 7):
        values = segment[:, signal][segment_covered]
        if values.size:
            output[offset] = float(values.mean())
            output[offset + 1] = float(values.std())
            output[offset + 2] = float(values[-1] - values[0])
        offset += 3
    return output


def temporal_features_for_rows(market: Mapping[str, object]) -> np.ndarray:
    """Reference implementation for one sample's raw market rows."""
    values = {name: np.asarray(market[name], dtype=float).reshape(-1) for name in _MARKET_COLUMNS[1:]}
    lengths = {value.size for value in values.values()}
    if len(lengths) != 1:
        raise ValueError("temporal market columns must have equal lengths")
    path, covered, updates = _asof_path(values)
    result = np.concatenate([_window_features(path, covered, updates if window == 60 else int(np.sum((values["seconds_before_predict"] >= 0) & (values["seconds_before_predict"] <= window))), window) for window in WINDOWS])
    if not np.isfinite(result).all():
        raise ValueError("temporal features must be finite")
    return result


def _stable_file_fingerprint(path: str | Path, sample_bytes: int = 1 << 20) -> str:
    target = Path(path)
    size = target.stat().st_size
    digest = hashlib.sha256(str(size).encode("ascii"))
    with target.open("rb") as handle:
        digest.update(handle.read(sample_bytes))
        if size > sample_bytes:
            handle.seek(max(0, size - sample_bytes))
            digest.update(handle.read(sample_bytes))
    return digest.hexdigest()


def _run_to_features(columns: Mapping[str, np.ndarray], start: int, end: int) -> np.ndarray:
    return temporal_features_for_rows({name: np.asarray(value)[start:end] for name, value in columns.items() if name != "sample_id"})


def _stream_temporal(path: str | Path, expected_ids: np.ndarray) -> tuple[np.ndarray, int, int]:
    try:
        import pyarrow.dataset as ds
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("PyArrow is required to build temporal Geometry features") from exc
    scanner = ds.dataset(str(path), format="ipc").scanner(columns=list(_MARKET_COLUMNS), batch_size=1_000_000)
    ids_out: list[int] = []
    features_out: list[np.ndarray] = []
    pending: dict[str, np.ndarray] | None = None
    pending_id: int | None = None
    invalid_rows = 0
    for batch in scanner.to_batches():
        columns = {name: batch.column(name).to_numpy(zero_copy_only=False) for name in _MARKET_COLUMNS}
        ids = np.asarray(columns.pop("sample_id"), dtype=np.int64)
        if ids.size == 0:
            continue
        boundaries = np.flatnonzero(ids[1:] != ids[:-1]) + 1
        starts = np.concatenate(([0], boundaries))
        ends = np.concatenate((boundaries, [ids.size]))
        runs = [(int(ids[end - 1]), start, end) for start, end in zip(starts, ends)]
        if pending_id is not None:
            if runs[0][0] == pending_id:
                merged = {name: np.concatenate((pending[name], np.asarray(columns[name])[runs[0][1]:runs[0][2]])) for name in columns}
                runs = runs[1:]
                if runs:
                    values = temporal_features_for_rows(merged)
                    ids_out.append(pending_id)
                    features_out.append(values)
                else:
                    pending = merged
                    continue
            else:
                ids_out.append(pending_id)
                features_out.append(temporal_features_for_rows(pending))
            pending = None
            pending_id = None
        if runs:
            for sample_id, start, end in runs[:-1]:
                values = {name: np.asarray(column)[start:end] for name, column in columns.items()}
                invalid_rows += int((~_valid_quote(values)).sum())
                ids_out.append(sample_id)
                features_out.append(temporal_features_for_rows(values))
            sample_id, start, end = runs[-1]
            pending_id = sample_id
            pending = {name: np.asarray(column)[start:end].copy() for name, column in columns.items()}
    if pending_id is not None and pending is not None:
        invalid_rows += int((~_valid_quote(pending)).sum())
        ids_out.append(pending_id)
        features_out.append(temporal_features_for_rows(pending))
    ids = np.asarray(ids_out, dtype=np.int64)
    if not np.array_equal(ids, expected_ids):
        raise ValueError("temporal market IDs do not exactly match the base Geometry IDs")
    return np.asarray(features_out, dtype=np.float32), invalid_rows, int(ids.size)


def build_geometry_temporal_file(
    base_path: str | Path,
    market_path: str | Path,
    output_path: str | Path,
) -> dict[str, object]:
    """Append fixed M02-T temporal features to the frozen M02-base artifact."""
    started = time.perf_counter()
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("PyArrow is required to build temporal Geometry features") from exc
    base_path = Path(base_path)
    manifest_path = base_path.parent / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError("M02-base manifest.json is required")
    base_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if base_manifest.get("experiment_id") != "m02-geometry-features" or base_manifest.get("status") != "complete":
        raise ValueError("M02-base manifest identity/status is invalid")
    base_names = tuple(base_manifest.get("diagnostics", {}).get("feature_names", ()))
    if base_names != geometry_feature_names() or len(base_names) != BASE_FEATURE_COUNT:
        raise ValueError("M02-base artifact must contain exactly the frozen 21 features")
    required = ("sample_id", "month", "target", *base_names)
    base = pq.read_table(base_path, columns=list(required))
    columns = {name: base[name].to_numpy(zero_copy_only=False) for name in required}
    order = np.argsort(columns["sample_id"], kind="mergesort")
    columns = {name: np.asarray(value)[order] for name, value in columns.items()}
    ids = columns["sample_id"].astype(np.int64)
    if np.unique(ids).size != ids.size or not np.isfinite(columns["target"]).all():
        raise ValueError("M02-base IDs must be unique and targets finite")
    temporal, invalid_rows, rows = _stream_temporal(market_path, ids)
    names = geometry_temporal_feature_names()
    values = np.column_stack([columns[name] for name in base_names] + [temporal[:, i] for i in range(temporal.shape[1])]).astype(np.float32)
    if values.shape != (ids.size, len(names)) or not np.isfinite(values).all():
        raise ValueError("M02-T output shape or finite checks failed")
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.table({"sample_id": ids, "month": columns["month"], "target": columns["target"], **{name: values[:, i] for i, name in enumerate(names)}}), output, compression="zstd", write_statistics=True)
    diagnostics = {
        "rows": int(rows), "months": [int(columns["month"].min()), int(columns["month"].max())],
        "feature_names": list(names), "feature_hash": feature_hash(list(names)),
        "base_feature_hash": base_manifest.get("feature_hash"),
        "base_artifact_hashes": base_manifest.get("diagnostics", {}).get("artifact_hashes", {}),
        "artifact_hashes": {"sample_id": array_hash(ids), "month": array_hash(columns["month"]), "target": array_hash(columns["target"]), "values": array_hash(values)},
        "invalid_quote_rows": int(invalid_rows), "temporal_feature_count": TEMPORAL_FEATURE_COUNT,
    }
    result_hash = hashlib.sha256(json.dumps(diagnostics, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    ExperimentManifest(
        experiment_id="m02-t-geometry-features", status="complete", config_hash=result_hash,
        data_fingerprints={"market.feather": _stable_file_fingerprint(market_path), "base_geometry.parquet": _stable_file_fingerprint(base_path)},
        feature_hash=diagnostics["feature_hash"], train_months=tuple(diagnostics["months"]), diagnostics=diagnostics,
        runtime_seconds=time.perf_counter() - started,
    ).write(output.parent)
    (output.parent / "report.md").write_text("\n".join([
        "# M02-T temporal Geometry features", "", f"- rows: `{rows}`", f"- features: `{len(names)}`",
        f"- invalid quote rows: `{invalid_rows}`", "- as-of grid: `1-second, 60s to 0s`",
        "- outer/test targets are not used while building features", "",
    ]), encoding="utf-8")
    return diagnostics | {"output": str(output), "result_hash": result_hash}
