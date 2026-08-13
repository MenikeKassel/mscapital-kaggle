"""M01-A signed Event Flow features for small arrays and large Feather files."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from ..artifacts import ExperimentManifest, feature_hash
from .ofi import WINDOWS, signed_order_flow, signed_trade_flow


def event_flow_feature_names(windows: Sequence[float] = WINDOWS) -> tuple[str, ...]:
    names: list[str] = []
    for source in ("order", "trade"):
        for window in windows:
            suffix = int(window)
            names.extend(
                [
                    f"{source}_signed_volume_per_second_{suffix}",
                    f"{source}_signed_volume_per_event_{suffix}",
                    f"{source}_event_count_per_second_{suffix}",
                ]
            )
    return tuple(names)


def _aggregate_arrays(
    sample_id: object,
    seconds_before_predict: object,
    signed_flow: object,
    *,
    source: str,
    windows: Sequence[float] = WINDOWS,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    ids = np.asarray(sample_id).reshape(-1)
    seconds = np.asarray(seconds_before_predict, dtype=np.float64).reshape(-1)
    flow = np.asarray(signed_flow, dtype=np.float64).reshape(-1)
    if not (ids.shape == seconds.shape == flow.shape):
        raise ValueError("sample_id, seconds_before_predict and signed flow must align")
    unique_ids = np.unique(ids)
    result = {
        name: np.zeros(unique_ids.size, dtype=np.float64)
        for name in event_flow_feature_names(windows)
        if name.startswith(f"{source}_")
    }
    for row, value in enumerate(unique_ids):
        sample = ids == value
        for window in windows:
            mask = sample & np.isfinite(seconds) & (seconds >= 0.0) & (seconds <= window)
            count = int(mask.sum())
            total = float(flow[mask].sum())
            suffix = int(window)
            result[f"{source}_signed_volume_per_second_{suffix}"][row] = total / window
            result[f"{source}_signed_volume_per_event_{suffix}"][row] = total / count if count else 0.0
            result[f"{source}_event_count_per_second_{suffix}"][row] = count / window
    return unique_ids, result


def build_event_flow_arrays(
    order: Mapping[str, object],
    transaction: Mapping[str, object],
    *,
    sample_ids: object | None = None,
    windows: Sequence[float] = WINDOWS,
) -> tuple[np.ndarray, list[str], np.ndarray]:
    """Reference M01-A implementation for tests and small samples."""

    order_flow = signed_order_flow(order["volume"], order["side"], order["order_action"])
    trade_flow = signed_trade_flow(transaction["volume"], transaction["side"])
    order_ids, order_features = _aggregate_arrays(
        order["sample_id"], order["seconds_before_predict"], order_flow,
        source="order", windows=windows,
    )
    trade_ids, trade_features = _aggregate_arrays(
        transaction["sample_id"], transaction["seconds_before_predict"], trade_flow,
        source="trade", windows=windows,
    )
    ids = (
        np.asarray(sample_ids).reshape(-1)
        if sample_ids is not None
        else np.union1d(order_ids, trade_ids)
    )
    if np.unique(ids).size != ids.size:
        raise ValueError("sample_ids must be unique")
    names = list(event_flow_feature_names(windows))
    values = np.zeros((ids.size, len(names)), dtype=np.float32)
    row_by_id = {value: row for row, value in enumerate(ids.tolist())}
    for source_ids, features in ((order_ids, order_features), (trade_ids, trade_features)):
        for source_row, sample_id in enumerate(source_ids.tolist()):
            target_row = row_by_id.get(sample_id)
            if target_row is None:
                continue
            for name, column in features.items():
                values[target_row, names.index(name)] = column[source_row]
    if not np.isfinite(values).all():
        raise ValueError("M01-A array features must be finite")
    return ids, names, values


def _source_lazy(path: str | Path, source: str):
    try:
        import polars as pl
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("polars is required for streaming M01-A aggregation") from exc

    frame = pl.scan_ipc(path)
    required = {"sample_id", "seconds_before_predict", "volume", "side"}
    if source == "order":
        required.add("order_action")
    missing = required - set(frame.collect_schema().names())
    if missing:
        raise ValueError(f"{source} Feather is missing columns: {sorted(missing)}")
    side_sign = pl.when(pl.col("side") == 0).then(1.0).otherwise(-1.0)
    signed = pl.col("volume").cast(pl.Float64) * side_sign
    if source == "order":
        action_sign = pl.when(pl.col("order_action") == 0).then(1.0).otherwise(-1.0)
        signed = signed * action_sign
    seconds = pl.col("seconds_before_predict").cast(pl.Float64)
    aggregations = []
    for window in WINDOWS:
        valid = seconds.is_finite() & (seconds >= 0.0) & (seconds <= window)
        count = valid.cast(pl.UInt32).sum()
        total = signed.filter(valid).sum()
        suffix = int(window)
        aggregations.extend(
            [
                (total / window).alias(f"{source}_signed_volume_per_second_{suffix}"),
                pl.when(count > 0)
                .then(total / count)
                .otherwise(0.0)
                .alias(f"{source}_signed_volume_per_event_{suffix}"),
                (count / window).alias(f"{source}_event_count_per_second_{suffix}"),
            ]
        )
    invalid = ~pl.col("side").is_in([0, 1])
    if source == "order":
        invalid = invalid | ~pl.col("order_action").is_in([0, 1])
    aggregations.append(invalid.cast(pl.UInt32).sum().alias("__invalid_code_count"))
    return frame.group_by("sample_id").agg(aggregations)


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


def build_event_flow_file(
    order_path: str | Path,
    transaction_path: str | Path,
    labels_path: str | Path,
    output_path: str | Path,
) -> dict[str, object]:
    """Stream raw events into one row per labeled sample and write Parquet."""

    try:
        import polars as pl
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("polars is required for streaming M01-A aggregation") from exc

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    order = _source_lazy(order_path, "order").collect(engine="streaming")
    transaction = _source_lazy(transaction_path, "trade").collect(engine="streaming")
    if int(order["__invalid_code_count"].sum()) or int(transaction["__invalid_code_count"].sum()):
        raise ValueError("order/trade side or action contains a code outside {0, 1}")
    order = order.drop("__invalid_code_count")
    transaction = transaction.drop("__invalid_code_count")
    labels = pl.read_ipc(labels_path).select("sample_id", "month", "target")
    if labels["sample_id"].n_unique() != labels.height:
        raise ValueError("labels sample_id must be unique")
    frame = (
        labels.join(order, on="sample_id", how="left", validate="1:1")
        .join(transaction, on="sample_id", how="left", validate="1:1")
        .fill_null(0.0)
        .sort("sample_id")
    )
    names = list(event_flow_feature_names())
    values = frame.select(names).to_numpy()
    if not np.isfinite(values).all():
        raise ValueError("M01-A streaming features must be finite")
    frame.write_parquet(output, compression="zstd", statistics=True)
    diagnostics = {
        "rows": frame.height,
        "months": [int(frame["month"].min()), int(frame["month"].max())],
        "feature_names": names,
        "feature_hash": feature_hash(names),
        "output_columns": frame.columns,
    }
    payload = json.dumps(diagnostics, sort_keys=True, separators=(",", ":")).encode("utf-8")
    result_hash = hashlib.sha256(payload).hexdigest()
    manifest = ExperimentManifest(
        experiment_id="m01-a-event-flow-features",
        status="complete",
        config_hash=result_hash,
        data_fingerprints={
            Path(path).name: _stable_file_fingerprint(path)
            for path in (order_path, transaction_path, labels_path)
        },
        feature_hash=diagnostics["feature_hash"],
        train_months=tuple(diagnostics["months"]),
        diagnostics=diagnostics,
    )
    manifest.write(output.parent)
    (output.parent / "report.md").write_text(
        "\n".join(
            [
                "# M01-A Event Flow features", "",
                f"- rows: `{frame.height}`",
                f"- features: `{len(names)}`",
                f"- months: `{diagnostics['months'][0]}-{diagnostics['months'][1]}`",
                "- aggregation: `Polars lazy streaming group_by`", "",
            ]
        ),
        encoding="utf-8",
    )
    return diagnostics | {"output": str(output), "result_hash": result_hash}
