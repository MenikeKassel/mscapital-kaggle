"""Fold-safe depth-2 path signatures for fixed L1/L2 quote paths.

The representation is deliberately frozen.  Seven channels are sampled on an
old-to-new one-second as-of grid and transformed into endpoint increments plus
the 21 antisymmetric level-two areas.  Every normalization is local to one
sample and one trailing window; labels and fold state never enter the feature
calculation.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Iterator, Mapping, Sequence

import numpy as np

from ..artifacts import ExperimentManifest, array_hash, feature_hash
from .ofi import quote_ofi


WINDOWS = (5, 15, 30, 60)
CHANNEL_NAMES = (
    "mid_log_return_bps",
    "relative_spread_bps",
    "l1_imbalance",
    "l2_imbalance",
    "normalized_l1_l2_cont_ofi",
    "normalized_signed_trade_volume",
    "normalized_event_clock",
)
FIRST_ORDER_COUNT = len(CHANNEL_NAMES)
SECOND_ORDER_COUNT = FIRST_ORDER_COUNT * (FIRST_ORDER_COUNT - 1) // 2
FEATURES_PER_WINDOW = FIRST_ORDER_COUNT + SECOND_ORDER_COUNT
PATH_SIGNATURE_FEATURE_COUNT = len(WINDOWS) * FEATURES_PER_WINDOW
STREAM_BATCH_SIZE = 1_000_000
_EPSILON = 1e-12

_MARKET_VALUE_COLUMNS = (
    "seconds_before_predict", "bid_price_1", "bid_volume_1",
    "ask_price_1", "ask_volume_1", "bid_price_2", "bid_volume_2",
    "ask_price_2", "ask_volume_2",
)
_ORDER_VALUE_COLUMNS = ("seconds_before_predict", "volume", "side", "order_action")
_TRADE_VALUE_COLUMNS = ("seconds_before_predict", "volume", "side")


def path_signature_feature_names() -> tuple[str, ...]:
    names: list[str] = []
    for window in WINDOWS:
        prefix = f"path_sig_t{window}"
        names.extend(f"{prefix}_level1_{channel}" for channel in CHANNEL_NAMES)
        names.extend(
            f"{prefix}_area_{CHANNEL_NAMES[left]}__{CHANNEL_NAMES[right]}"
            for left in range(FIRST_ORDER_COUNT)
            for right in range(left + 1, FIRST_ORDER_COUNT)
        )
    return tuple(names)


def depth2_path_signature(path: object) -> np.ndarray:
    """Return endpoint increments and centered signed areas for a path.

    At least three observed points are required.  Centering at the first point
    makes the shoelace calculation translation invariant and equal to the
    antisymmetric component of the piecewise-linear depth-two signature.
    """

    values = np.asarray(path, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != FIRST_ORDER_COUNT:
        raise ValueError(f"signature path must have shape (n, {FIRST_ORDER_COUNT})")
    if values.shape[0] < 3:
        return np.zeros(FEATURES_PER_WINDOW, dtype=np.float32)
    if not np.isfinite(values).all():
        raise ValueError("signature path must be finite")
    centered = values - values[0]
    output = np.empty(FEATURES_PER_WINDOW, dtype=np.float64)
    output[:FIRST_ORDER_COUNT] = centered[-1]
    offset = FIRST_ORDER_COUNT
    for left in range(FIRST_ORDER_COUNT):
        for right in range(left + 1, FIRST_ORDER_COUNT):
            output[offset] = 0.5 * float(np.sum(
                centered[:-1, left] * centered[1:, right]
                - centered[:-1, right] * centered[1:, left]
            ))
            offset += 1
    if not np.isfinite(output).all():
        raise ValueError("signature features must be finite")
    return output.astype(np.float32)


def _equal_length_arrays(rows: Mapping[str, object], columns: Sequence[str], source: str) -> dict[str, np.ndarray]:
    missing = set(columns) - set(rows)
    if missing:
        raise ValueError(f"{source} rows are missing columns: {sorted(missing)}")
    values = {name: np.asarray(rows[name]).reshape(-1) for name in columns}
    if len({value.size for value in values.values()}) != 1:
        raise ValueError(f"{source} columns must have equal lengths")
    return values


def _valid_quotes(rows: Mapping[str, np.ndarray]) -> np.ndarray:
    seconds = np.asarray(rows["seconds_before_predict"], dtype=float)
    bid1 = np.asarray(rows["bid_price_1"], dtype=float)
    ask1 = np.asarray(rows["ask_price_1"], dtype=float)
    bid2 = np.asarray(rows["bid_price_2"], dtype=float)
    ask2 = np.asarray(rows["ask_price_2"], dtype=float)
    volumes = tuple(np.asarray(rows[name], dtype=float) for name in (
        "bid_volume_1", "ask_volume_1", "bid_volume_2", "ask_volume_2"
    ))
    valid = np.isfinite(seconds) & (seconds >= 0.0) & (seconds <= 60.0)
    valid &= np.isfinite(bid1) & np.isfinite(ask1) & np.isfinite(bid2) & np.isfinite(ask2)
    valid &= (bid2 <= bid1) & (bid1 < ask1) & (ask1 <= ask2)
    for volume in volumes:
        valid &= np.isfinite(volume) & (volume >= 0.0)
    return valid


def _valid_orders(rows: Mapping[str, np.ndarray]) -> np.ndarray:
    seconds = np.asarray(rows["seconds_before_predict"], dtype=float)
    volume = np.asarray(rows["volume"], dtype=float)
    side = np.asarray(rows["side"])
    action = np.asarray(rows["order_action"])
    return (
        np.isfinite(seconds) & (seconds >= 0.0) & (seconds <= 60.0)
        & np.isfinite(volume) & (volume >= 0.0)
        & np.isin(side, (0, 1)) & np.isin(action, (0, 1))
    )


def _valid_trades(rows: Mapping[str, np.ndarray]) -> np.ndarray:
    seconds = np.asarray(rows["seconds_before_predict"], dtype=float)
    volume = np.asarray(rows["volume"], dtype=float)
    side = np.asarray(rows["side"])
    return (
        np.isfinite(seconds) & (seconds >= 0.0) & (seconds <= 60.0)
        & np.isfinite(volume) & (volume >= 0.0) & np.isin(side, (0, 1))
    )


def _quote_ofi(rows: Mapping[str, np.ndarray]) -> np.ndarray:
    return quote_ofi(
        rows["bid_price_1"], rows["bid_volume_1"],
        rows["ask_price_1"], rows["ask_volume_1"],
    ) + quote_ofi(
        rows["bid_price_2"], rows["bid_volume_2"],
        rows["ask_price_2"], rows["ask_volume_2"],
    )


def _window_path(
    quote_rows: Mapping[str, np.ndarray],
    trade_rows: Mapping[str, np.ndarray],
    window: int,
) -> tuple[np.ndarray, np.ndarray]:
    grid = np.arange(float(window), -1.0, -1.0)
    path = np.zeros((grid.size, FIRST_ORDER_COUNT), dtype=np.float64)
    covered = np.zeros(grid.size, dtype=bool)
    if quote_rows["seconds_before_predict"].size == 0:
        return path, covered

    quote_seconds = quote_rows["seconds_before_predict"]
    quote_ofi = np.cumsum(_quote_ofi(quote_rows))
    quote_at_grid = np.full(grid.size, -1, dtype=np.int64)
    position = -1
    for row, second in enumerate(grid):
        while position + 1 < quote_seconds.size and quote_seconds[position + 1] >= second:
            position += 1
        if position >= 0:
            quote_at_grid[row] = position
            covered[row] = True
    if int(covered.sum()) < 3:
        return path, covered

    selected = quote_at_grid[covered]
    bid1 = quote_rows["bid_price_1"][selected]
    ask1 = quote_rows["ask_price_1"][selected]
    mid = (bid1 + ask1) / 2.0
    path[covered, 0] = np.log(mid / mid[0]) * 1e4
    path[covered, 1] = (ask1 - bid1) / np.maximum(np.abs(mid), _EPSILON) * 1e4
    for column, bid_name, ask_name in (
        (2, "bid_volume_1", "ask_volume_1"),
        (3, "bid_volume_2", "ask_volume_2"),
    ):
        bid_volume = quote_rows[bid_name][selected]
        ask_volume = quote_rows[ask_name][selected]
        path[covered, column] = (bid_volume - ask_volume) / (bid_volume + ask_volume + _EPSILON)

    quote_window = quote_rows["seconds_before_predict"] <= float(window)
    total_depth = sum(
        quote_rows[name][quote_window]
        for name in ("bid_volume_1", "ask_volume_1", "bid_volume_2", "ask_volume_2")
    )
    depth_scale = max(float(np.mean(total_depth)) if total_depth.size else 0.0, _EPSILON)
    first_quote = int(selected[0])
    path[covered, 4] = (quote_ofi[selected] - quote_ofi[first_quote]) / depth_scale

    trade_seconds = trade_rows["seconds_before_predict"]
    in_window = trade_seconds <= float(window)
    if in_window.any():
        seconds = trade_seconds[in_window]
        volumes = trade_rows["volume"][in_window]
        sides = trade_rows["side"][in_window]
        signed = volumes * np.where(sides == 0, 1.0, -1.0)
        cumulative = np.cumsum(signed)
        scale = max(float(np.sum(np.abs(volumes))), _EPSILON)
        trade_position = -1
        for row, second in enumerate(grid):
            while trade_position + 1 < seconds.size and seconds[trade_position + 1] >= second:
                trade_position += 1
            if covered[row] and trade_position >= 0:
                path[row, 5] = cumulative[trade_position] / scale

    path[covered, 6] = (float(window) - grid[covered]) / float(window)
    return path, covered


def path_signature_features_for_rows(
    market: Mapping[str, object],
    order: Mapping[str, object],
    transaction: Mapping[str, object],
) -> np.ndarray:
    """Reference builder for one sample's raw rows."""

    market_rows = _equal_length_arrays(market, _MARKET_VALUE_COLUMNS, "market")
    order_rows = _equal_length_arrays(order, _ORDER_VALUE_COLUMNS, "order")
    trade_rows = _equal_length_arrays(transaction, _TRADE_VALUE_COLUMNS, "transaction")

    valid_quote = _valid_quotes(market_rows)
    quote_rows = {name: np.asarray(value[valid_quote], dtype=np.float64) for name, value in market_rows.items()}
    if quote_rows["seconds_before_predict"].size:
        quote_order = np.argsort(-quote_rows["seconds_before_predict"], kind="mergesort")
        quote_rows = {name: value[quote_order] for name, value in quote_rows.items()}

    # Order events are part of the frozen raw-input contract and are validated
    # for diagnostics, but none of the seven named channels depends on them.
    _valid_orders(order_rows)
    valid_trade = _valid_trades(trade_rows)
    trade_rows = {name: np.asarray(value[valid_trade], dtype=np.float64) for name, value in trade_rows.items()}
    if trade_rows["seconds_before_predict"].size:
        trade_order = np.argsort(-trade_rows["seconds_before_predict"], kind="mergesort")
        trade_rows = {name: value[trade_order] for name, value in trade_rows.items()}

    features: list[np.ndarray] = []
    for window in WINDOWS:
        path, covered = _window_path(quote_rows, trade_rows, window)
        features.append(depth2_path_signature(path[covered]))
    result = np.concatenate(features).astype(np.float32)
    if result.shape != (PATH_SIGNATURE_FEATURE_COUNT,) or not np.isfinite(result).all():
        raise ValueError("M03 path-signature output shape or finite check failed")
    return result


def _empty(columns: Sequence[str]) -> dict[str, np.ndarray]:
    return {name: np.asarray([], dtype=np.float64) for name in columns}


def build_path_signature_arrays(
    market: Mapping[str, object],
    order: Mapping[str, object],
    transaction: Mapping[str, object],
    *,
    sample_ids: object | None = None,
) -> tuple[np.ndarray, tuple[str, ...], np.ndarray]:
    """Small-array reference builder used to verify the streaming path."""

    sources = []
    for rows, columns, source in (
        (market, _MARKET_VALUE_COLUMNS, "market"),
        (order, _ORDER_VALUE_COLUMNS, "order"),
        (transaction, _TRADE_VALUE_COLUMNS, "transaction"),
    ):
        values = _equal_length_arrays(rows, ("sample_id", *columns), source)
        sources.append(values)
    if sample_ids is None:
        ids = np.unique(np.concatenate([source["sample_id"] for source in sources])).astype(np.int64)
    else:
        ids = np.asarray(sample_ids, dtype=np.int64).reshape(-1)
    if np.unique(ids).size != ids.size:
        raise ValueError("sample_ids must be unique")
    matrix = np.zeros((ids.size, PATH_SIGNATURE_FEATURE_COUNT), dtype=np.float32)
    for row, sample_id in enumerate(ids):
        grouped = []
        for source, columns in zip(sources, (_MARKET_VALUE_COLUMNS, _ORDER_VALUE_COLUMNS, _TRADE_VALUE_COLUMNS)):
            mask = source["sample_id"] == sample_id
            grouped.append({name: source[name][mask] for name in columns})
        matrix[row] = path_signature_features_for_rows(*grouped)
    return ids, path_signature_feature_names(), matrix


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


def _grouped_ipc_rows(path: str | Path, columns: Sequence[str]) -> Iterator[tuple[int, dict[str, np.ndarray]]]:
    try:
        import pyarrow.dataset as ds
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("PyArrow is required to build M03 path signatures") from exc
    scanner = ds.dataset(str(path), format="ipc").scanner(
        columns=["sample_id", *columns], batch_size=STREAM_BATCH_SIZE
    )
    pending_id: int | None = None
    pending: dict[str, list[np.ndarray]] = {name: [] for name in columns}
    last_id: int | None = None
    for batch in scanner.to_batches():
        values = {
            name: batch.column(name).to_numpy(zero_copy_only=False)
            for name in ("sample_id", *columns)
        }
        ids = np.asarray(values.pop("sample_id"), dtype=np.int64)
        if ids.size == 0:
            continue
        if np.any(ids[1:] < ids[:-1]) or (last_id is not None and ids[0] < last_id):
            raise ValueError(f"{Path(path).name} must be sorted by nondecreasing sample_id")
        boundaries = np.flatnonzero(ids[1:] != ids[:-1]) + 1
        starts = np.concatenate(([0], boundaries))
        ends = np.concatenate((boundaries, [ids.size]))
        for start, end in zip(starts, ends):
            sample_id = int(ids[start])
            if pending_id is not None and sample_id != pending_id:
                yield pending_id, {name: np.concatenate(parts) for name, parts in pending.items()}
                pending = {name: [] for name in columns}
            pending_id = sample_id
            for name in columns:
                pending[name].append(np.asarray(values[name][start:end]))
        last_id = int(ids[-1])
    if pending_id is not None:
        yield pending_id, {name: np.concatenate(parts) for name, parts in pending.items()}


def _take_group(
    current: tuple[int, dict[str, np.ndarray]] | None,
    iterator: Iterator[tuple[int, dict[str, np.ndarray]]],
    sample_id: int,
    empty_columns: Sequence[str],
) -> tuple[dict[str, np.ndarray], tuple[int, dict[str, np.ndarray]] | None]:
    if current is not None and current[0] < sample_id:
        raise ValueError(f"raw sample_id {current[0]} is absent from labels")
    if current is not None and current[0] == sample_id:
        rows = current[1]
        return rows, next(iterator, None)
    return _empty(empty_columns), current


def build_path_signature_file(
    market_path: str | Path,
    order_path: str | Path,
    transaction_path: str | Path,
    labels_path: str | Path,
    output_path: str | Path,
) -> dict[str, object]:
    """Stream raw grouped Feather inputs into one fixed M03 row per label."""

    started = time.perf_counter()
    try:
        import pyarrow as pa
        import pyarrow.feather as feather
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("PyArrow is required to build M03 path signatures") from exc

    labels = feather.read_table(str(labels_path), columns=["sample_id", "month", "target"])
    label_ids = labels["sample_id"].to_numpy(zero_copy_only=False).astype(np.int64)
    months = labels["month"].to_numpy(zero_copy_only=False)
    targets = labels["target"].to_numpy(zero_copy_only=False)
    label_order = np.argsort(label_ids, kind="mergesort")
    label_ids, months, targets = label_ids[label_order], months[label_order], targets[label_order]
    if np.unique(label_ids).size != label_ids.size or not np.isfinite(targets).all():
        raise ValueError("labels must have unique IDs and finite targets")

    iterators = (
        iter(_grouped_ipc_rows(market_path, _MARKET_VALUE_COLUMNS)),
        iter(_grouped_ipc_rows(order_path, _ORDER_VALUE_COLUMNS)),
        iter(_grouped_ipc_rows(transaction_path, _TRADE_VALUE_COLUMNS)),
    )
    current = [next(iterator, None) for iterator in iterators]
    values = np.zeros((label_ids.size, PATH_SIGNATURE_FEATURE_COUNT), dtype=np.float32)
    invalid_quote_rows = invalid_order_rows = invalid_trade_rows = 0
    for row, sample_id in enumerate(label_ids):
        grouped = []
        for index, columns in enumerate((_MARKET_VALUE_COLUMNS, _ORDER_VALUE_COLUMNS, _TRADE_VALUE_COLUMNS)):
            rows, current[index] = _take_group(current[index], iterators[index], int(sample_id), columns)
            grouped.append(rows)
        invalid_quote_rows += int((~_valid_quotes(grouped[0])).sum())
        invalid_order_rows += int((~_valid_orders(grouped[1])).sum())
        invalid_trade_rows += int((~_valid_trades(grouped[2])).sum())
        values[row] = path_signature_features_for_rows(*grouped)
    if any(item is not None for item in current):
        extra = next(item[0] for item in current if item is not None)
        raise ValueError(f"raw sample_id {extra} is absent from labels")
    if not np.isfinite(values).all():
        raise ValueError("M03 streamed features must be finite")

    names = path_signature_feature_names()
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.table({
        "sample_id": label_ids, "month": months, "target": targets,
        **{name: values[:, index] for index, name in enumerate(names)},
    }), output, compression="zstd", write_statistics=True)
    diagnostics = {
        "rows": int(label_ids.size),
        "months": [int(np.min(months)), int(np.max(months))],
        "feature_names": list(names),
        "feature_hash": feature_hash(list(names)),
        "feature_count": PATH_SIGNATURE_FEATURE_COUNT,
        "invalid_quote_rows": invalid_quote_rows,
        "invalid_order_rows": invalid_order_rows,
        "invalid_trade_rows": invalid_trade_rows,
        "artifact_hashes": {
            "sample_id": array_hash(label_ids), "month": array_hash(months),
            "target": array_hash(targets), "values": array_hash(values),
        },
    }
    result_hash = hashlib.sha256(
        json.dumps(diagnostics, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    ExperimentManifest(
        experiment_id="m03-path-signature-features", status="complete",
        config_hash=result_hash,
        data_fingerprints={
            "market.feather": _stable_file_fingerprint(market_path),
            "order.feather": _stable_file_fingerprint(order_path),
            "transaction.feather": _stable_file_fingerprint(transaction_path),
            "label.feather": _stable_file_fingerprint(labels_path),
        },
        feature_hash=diagnostics["feature_hash"], train_months=tuple(diagnostics["months"]),
        diagnostics=diagnostics, runtime_seconds=time.perf_counter() - started,
    ).write(output.parent)
    (output.parent / "report.md").write_text("\n".join([
        "# M03 depth-2 Path Signature features", "",
        f"- rows: `{label_ids.size}`", f"- features: `{len(names)}`",
        "- channels: `7 fixed, window-local`", "- windows: `5 / 15 / 30 / 60 seconds`",
        "- grid: `1-second, old-to-new as-of`", "- labels/folds are not used to build features", "",
    ]), encoding="utf-8")
    return diagnostics | {"output": str(output), "result_hash": result_hash}
