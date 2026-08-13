"""Market-centered L1/L2 order-book geometry."""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Mapping

import numpy as np

from ..artifacts import ExperimentManifest, array_hash, feature_hash


_MARKET_COLUMNS = (
    "sample_id", "seconds_before_predict", "bid_price_1", "bid_volume_1",
    "ask_price_1", "ask_volume_1", "bid_price_2", "bid_volume_2",
    "ask_price_2", "ask_volume_2",
)


def _entropy(values: np.ndarray) -> float:
    total = float(values.sum())
    if total <= 0:
        return 0.0
    p = values / total
    return float(-(p[p > 0] * np.log(p[p > 0])).sum())


def lob_geometry_row(
    bid_price_1: float,
    bid_volume_1: float,
    ask_price_1: float,
    ask_volume_1: float,
    bid_price_2: float,
    bid_volume_2: float,
    ask_price_2: float,
    ask_volume_2: float,
) -> dict[str, float]:
    mid = (bid_price_1 + ask_price_1) / 2.0
    spread = max(ask_price_1 - bid_price_1, 1e-12)
    depths = np.array([bid_volume_1, ask_volume_1, bid_volume_2, ask_volume_2], dtype=float)
    total = max(float(depths.sum()), 1e-12)
    bid_share = (bid_volume_1 + bid_volume_2) / total
    ask_share = (ask_volume_1 + ask_volume_2) / total
    bid_near_far = bid_volume_1 / max(bid_volume_2, 1e-12)
    ask_near_far = ask_volume_1 / max(ask_volume_2, 1e-12)
    return {
        "lob_quote_missing": 0.0,
        "lob_l1_depth_share": (bid_volume_1 + ask_volume_1) / total,
        "lob_bid_l1_l2_gap": (bid_price_1 - bid_price_2) / spread,
        "lob_ask_l1_l2_gap": (ask_price_2 - ask_price_1) / spread,
        # L1 relative prices are exactly +/- 0.5 by construction and omitted.
        "lob_bid2_rel_mid_spread": (bid_price_2 - mid) / spread,
        "lob_ask2_rel_mid_spread": (ask_price_2 - mid) / spread,
        "lob_bid1_depth_share": bid_volume_1 / total,
        "lob_ask1_depth_share": ask_volume_1 / total,
        "lob_bid2_depth_share": bid_volume_2 / total,
        "lob_ask2_depth_share": ask_volume_2 / total,
        "lob_bid_depth_share": bid_share,
        "lob_ask_depth_share": ask_share,
        "lob_near_far_bid": bid_near_far,
        "lob_near_far_ask": ask_near_far,
        "lob_depth_slope_bid": bid_volume_2 - bid_volume_1,
        "lob_depth_slope_ask": ask_volume_2 - ask_volume_1,
        "lob_depth_entropy": _entropy(depths),
        "lob_depth_hhi": float(np.square(depths / total).sum()),
        "lob_shape_asymmetry": bid_share - ask_share,
        "lob_slope_asymmetry": (bid_volume_2 - bid_volume_1) - (ask_volume_2 - ask_volume_1),
        "lob_l1_l2_volume_ratio": (bid_volume_1 + ask_volume_1) / max(bid_volume_2 + ask_volume_2, 1e-12),
    }


def geometry_feature_names() -> tuple[str, ...]:
    return tuple(sorted(lob_geometry_row(99, 10, 101, 20, 98, 5, 102, 15)))


def build_lob_geometry(market: Mapping[str, object]) -> tuple[np.ndarray, list[str], np.ndarray]:
    sample_id = np.asarray(market["sample_id"], dtype=np.int64).reshape(-1)
    if sample_id.size == 0 or np.any(sample_id[1:] < sample_id[:-1]):
        raise ValueError("market sample_id must be sorted and non-empty")
    boundaries = np.flatnonzero(sample_id[1:] != sample_id[:-1]) + 1
    indices = np.concatenate((boundaries - 1, [sample_id.size - 1]))
    if "seconds_before_predict" in market:
        seconds = np.asarray(market["seconds_before_predict"], dtype=float).reshape(-1)
        starts = np.concatenate(([0], boundaries))
        ends = np.concatenate((boundaries, [sample_id.size]))
        # The source is chronological oldest-to-newest within each sample, so the
        # final row is the latest quote (smallest seconds-before-predict).  Keep
        # the check vectorized: the common path must not iterate 1.25M samples in
        # Python.  A non-monotone source falls back to the exact per-group argmin.
        minima = np.minimum.reduceat(seconds, starts)
        final_seconds = seconds[indices]
        if np.any(minima < final_seconds):
            indices = np.array([
                start + int(np.nanargmin(seconds[start:end]))
                for start, end in zip(starts, ends)
            ], dtype=np.int64)
    ids = sample_id[indices]
    bid1 = np.asarray(market["bid_price_1"], dtype=float).reshape(-1)[indices]
    ask1 = np.asarray(market["ask_price_1"], dtype=float).reshape(-1)[indices]
    bid2 = np.asarray(market["bid_price_2"], dtype=float).reshape(-1)[indices]
    ask2 = np.asarray(market["ask_price_2"], dtype=float).reshape(-1)[indices]
    bv1 = np.asarray(market["bid_volume_1"], dtype=float).reshape(-1)[indices]
    av1 = np.asarray(market["ask_volume_1"], dtype=float).reshape(-1)[indices]
    bv2 = np.asarray(market["bid_volume_2"], dtype=float).reshape(-1)[indices]
    av2 = np.asarray(market["ask_volume_2"], dtype=float).reshape(-1)[indices]
    mid = (bid1 + ask1) / 2.0
    spread = np.maximum(ask1 - bid1, 1e-12)
    total = np.maximum(bv1 + av1 + bv2 + av2, 1e-12)
    raw = {
        "lob_quote_missing": np.zeros(ids.size, dtype=np.float64),
        "lob_l1_depth_share": (bv1 + av1) / total,
        "lob_bid_l1_l2_gap": (bid1 - bid2) / spread,
        "lob_ask_l1_l2_gap": (ask2 - ask1) / spread,
        "lob_bid2_rel_mid_spread": (bid2 - mid) / spread,
        "lob_ask2_rel_mid_spread": (ask2 - mid) / spread,
        "lob_bid1_depth_share": bv1 / total, "lob_ask1_depth_share": av1 / total,
        "lob_bid2_depth_share": bv2 / total, "lob_ask2_depth_share": av2 / total,
        "lob_bid_depth_share": (bv1 + bv2) / total,
        "lob_ask_depth_share": (av1 + av2) / total,
        "lob_near_far_bid": bv1 / np.maximum(bv2, 1e-12),
        "lob_near_far_ask": av1 / np.maximum(av2, 1e-12),
        "lob_depth_slope_bid": bv2 - bv1, "lob_depth_slope_ask": av2 - av1,
        "lob_depth_entropy": _entropy_vectorized(np.stack((bv1, av1, bv2, av2), axis=1)),
        "lob_depth_hhi": ((bv1 / total) ** 2 + (av1 / total) ** 2 + (bv2 / total) ** 2 + (av2 / total) ** 2),
        "lob_shape_asymmetry": (bv1 + bv2 - av1 - av2) / total,
        "lob_slope_asymmetry": (bv2 - bv1) - (av2 - av1),
        "lob_l1_l2_volume_ratio": (bv1 + av1) / np.maximum(bv2 + av2, 1e-12),
    }
    names = sorted(raw)
    values = np.column_stack([raw[name] for name in names]).astype(np.float32)
    return ids, names, values


def _entropy_vectorized(values: np.ndarray) -> np.ndarray:
    total = np.maximum(values.sum(axis=1), 1e-12)
    probabilities = values / total[:, None]
    terms = np.where(probabilities > 0, probabilities * np.log(np.maximum(probabilities, 1e-12)), 0.0)
    return -terms.sum(axis=1)


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


def _latest_quotes_stream(path: str | Path) -> dict[str, np.ndarray]:
    """Read a sorted market Feather file without materializing all 222M rows."""
    try:
        import pyarrow.dataset as ds
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("PyArrow is required to build M02 Geometry features") from exc
    scanner = ds.dataset(str(path), format="ipc").scanner(
        columns=list(_MARKET_COLUMNS), batch_size=1_000_000
    )
    ids_out: list[np.ndarray] = []
    rows_out: dict[str, list[np.ndarray]] = {name: [] for name in _MARKET_COLUMNS[1:]}
    pending_id: int | None = None
    pending_values: dict[str, np.generic] | None = None
    last_seen_id: int | None = None
    for batch in scanner.to_batches():
        columns = {
            name: batch.column(name).to_numpy(zero_copy_only=False)
            for name in _MARKET_COLUMNS
        }
        ids = np.asarray(columns["sample_id"], dtype=np.int64)
        if ids.size == 0:
            continue
        if np.any(ids[1:] < ids[:-1]) or (last_seen_id is not None and ids[0] < last_seen_id):
            raise ValueError("market Feather must be sorted by nondecreasing sample_id")
        boundaries = np.flatnonzero(ids[1:] != ids[:-1]) + 1
        starts = np.concatenate(([0], boundaries))
        ends = np.concatenate((boundaries, [ids.size]))
        # Select the latest usable timestamp per run.  Source rows are normally
        # ordered, but this keeps the streaming path correct if a batch is not.
        seconds = np.asarray(columns["seconds_before_predict"], dtype=np.float64)
        run_ids = ids[ends - 1]
        latest_indices = ends - 1
        minima = np.minimum.reduceat(seconds, starts)
        if np.any(minima < seconds[latest_indices]):
            latest_indices = np.asarray([
                start + int(np.nanargmin(seconds[start:end]))
                for start, end in zip(starts, ends)
            ], dtype=np.int64)
        run_values = {name: np.asarray(columns[name])[latest_indices] for name in _MARKET_COLUMNS[1:]}
        if pending_id is not None:
            if int(run_ids[0]) == pending_id:
                # The first run continues across the batch boundary; its final
                # row supersedes the pending row from the previous batch.
                pending_values = {
                    name: np.asarray(run_values[name][0]).item()
                    for name in _MARKET_COLUMNS[1:]
                }
                run_ids = run_ids[1:]
                run_values = {
                    name: value[1:] for name, value in run_values.items()
                }
                # The continued run is complete once this batch contributes a
                # different sample.  Flush the updated pending row before
                # appending the new runs; otherwise it is silently overwritten
                # by the final run of this batch.
                if run_ids.size:
                    ids_out.append(np.asarray([pending_id], dtype=np.int64))
                    for name in _MARKET_COLUMNS[1:]:
                        rows_out[name].append(np.asarray([pending_values[name]]))
                    pending_id = None
                    pending_values = None
            else:
                ids_out.append(np.asarray([pending_id], dtype=np.int64))
                for name in _MARKET_COLUMNS[1:]:
                    rows_out[name].append(np.asarray([pending_values[name]]))
                pending_id = None
                pending_values = None
        if run_ids.size:
            if pending_id is not None and int(run_ids[0]) <= pending_id:
                raise ValueError("market sample_id groups must be contiguous")
            if run_ids.size > 1:
                ids_out.append(run_ids[:-1].astype(np.int64, copy=False))
                for name in _MARKET_COLUMNS[1:]:
                    rows_out[name].append(run_values[name][:-1])
            pending_id = int(run_ids[-1])
            pending_values = {
                name: np.asarray(run_values[name][-1]).item()
                for name in _MARKET_COLUMNS[1:]
            }
        last_seen_id = int(ids[-1])
    if pending_id is not None:
        ids_out.append(np.asarray([pending_id], dtype=np.int64))
        for name in _MARKET_COLUMNS[1:]:
            rows_out[name].append(np.asarray([pending_values[name]]))
    if not ids_out:
        raise ValueError("market Feather contains no rows")
    return {"sample_id": np.concatenate(ids_out), **{
        name: np.concatenate(values) for name, values in rows_out.items()
    }}


def build_lob_geometry_file(
    market_path: str | Path,
    labels_path: str | Path,
    output_path: str | Path,
) -> dict[str, object]:
    """Build one latest-quote Geometry row per labeled sample and write Parquet."""
    started = time.perf_counter()
    try:
        import pyarrow as pa
        import pyarrow.feather as feather
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("PyArrow is required to build M02 Geometry features") from exc
    labels = feather.read_table(str(labels_path), columns=["sample_id", "month", "target"])
    label_ids = labels["sample_id"].to_numpy(zero_copy_only=False).astype(np.int64)
    label_month = labels["month"].to_numpy(zero_copy_only=False).astype(np.int16)
    label_target = labels["target"].to_numpy(zero_copy_only=False).astype(np.float64)
    label_order = np.argsort(label_ids, kind="mergesort")
    label_ids, label_month, label_target = (
        label_ids[label_order], label_month[label_order], label_target[label_order]
    )
    if np.unique(label_ids).size != label_ids.size or not np.isfinite(label_target).all():
        raise ValueError("labels must have unique IDs and finite targets")
    market = _latest_quotes_stream(market_path)
    ids, names, values = build_lob_geometry(market)
    if np.any(np.searchsorted(label_ids, ids) >= label_ids.size) or not np.array_equal(
        label_ids[np.searchsorted(label_ids, ids)], ids
    ):
        raise ValueError("market latest-quote IDs must be contained in labels")
    # A small number of labelled samples can have no quote rows.  Preserve the
    # labelled row and expose the absence explicitly so fold-local preprocessing
    # can learn its treatment without silently dropping observations.
    full_values = np.zeros((label_ids.size, len(names)), dtype=np.float32)
    full_values[:, names.index("lob_quote_missing")] = 1.0
    positions = np.searchsorted(label_ids, ids)
    full_values[positions] = values
    full_values[positions, names.index("lob_quote_missing")] = 0.0
    finite_rows = np.isfinite(full_values).all(axis=1)
    if not np.all(finite_rows):
        full_values[~finite_rows] = 0.0
        full_values[~finite_rows, names.index("lob_quote_missing")] = 1.0
    ids = label_ids
    values = full_values
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    table = pa.table({"sample_id": ids, "month": label_month, "target": label_target, **{
        name: values[:, index] for index, name in enumerate(names)
    }})
    pq.write_table(table, output, compression="zstd", write_statistics=True)
    diagnostics = {
        "rows": int(ids.size), "months": [int(label_month.min()), int(label_month.max())],
        "feature_names": names, "feature_hash": feature_hash(names),
        "artifact_hashes": {
            "sample_id": array_hash(ids), "month": array_hash(label_month),
            "target": array_hash(label_target), "values": array_hash(values),
        },
    }
    result_hash = hashlib.sha256(
        json.dumps(diagnostics, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    ExperimentManifest(
        experiment_id="m02-geometry-features", status="complete", config_hash=result_hash,
        data_fingerprints={
            "market.feather": _stable_file_fingerprint(market_path),
            "label.feather": _stable_file_fingerprint(labels_path),
        }, feature_hash=diagnostics["feature_hash"],
        train_months=tuple(diagnostics["months"]), diagnostics=diagnostics,
        runtime_seconds=time.perf_counter() - started,
    ).write(output.parent)
    (output.parent / "report.md").write_text(
        "\n".join([
            "# M02 Market-Centered LOB Geometry features", "",
            f"- rows: `{ids.size}`", f"- features: `{len(names)}`",
            f"- months: `{diagnostics['months'][0]}-{diagnostics['months'][1]}`",
            "- source: `latest quote per sample from L1/L2 market Feather`",
            "- L1 relative prices and L3+ curvature: `not included`", "",
        ]), encoding="utf-8"
    )
    return diagnostics | {"output": str(output), "result_hash": result_hash}
