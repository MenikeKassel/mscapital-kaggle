"""Fixed Optiver-style interactions over frozen M02-base and M01-A inputs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol
import hashlib
import json

import numpy as np

from ..artifacts import ExperimentManifest, array_hash, feature_hash


EPSILON = 1e-12
M04_FEATURE_COUNT = 24
_WINDOWS = (5, 15, 30, 60)


class _FeatureFrame(Protocol):
    sample_id: np.ndarray
    month: np.ndarray
    target: np.ndarray
    values: np.ndarray
    feature_names: tuple[str, ...]


@dataclass(frozen=True)
class OptiverInteractionFrame:
    sample_id: np.ndarray
    month: np.ndarray
    target: np.ndarray
    values: np.ndarray
    feature_names: tuple[str, ...]

    def validate(self) -> None:
        n = np.asarray(self.sample_id).reshape(-1).size
        if any(np.asarray(value).reshape(-1).size != n for value in (self.month, self.target)):
            raise ValueError("M04 identifiers, months and targets must align")
        if np.asarray(self.values).shape != (n, M04_FEATURE_COUNT):
            raise ValueError("M04 matrix must contain exactly 24 features")
        if self.feature_names != optiver_interaction_feature_names():
            raise ValueError("M04 feature schema is not the frozen interaction family")
        if np.unique(self.sample_id).size != n:
            raise ValueError("M04 sample_id must be unique")
        if not np.isfinite(self.target).all() or not np.isfinite(self.values).all():
            raise ValueError("M04 targets and features must be finite")


def optiver_interaction_feature_names() -> tuple[str, ...]:
    triplets = (
        "bid1_share_bid2_share_ask1_share",
        "bid1_share_bid2_share_ask2_share",
        "ask1_share_ask2_share_bid1_share",
        "ask1_share_ask2_share_bid2_share",
        "l1_imbalance_l2_imbalance_book_imbalance",
        "bid_gap_ask_gap_l1_l2_volume_ratio",
    )
    urgency = (
        "l1_imbalance", "l2_imbalance", "book_imbalance",
        "order_pressure_5", "trade_pressure_5", "order_trade_agreement_5",
    )
    depth = (
        "l1_imbalance_bid_gap", "l2_imbalance_ask_gap",
        "book_imbalance_mean_gap", "shape_asymmetry_book_imbalance",
    )
    flow = tuple(
        f"flow_combined_pressure_{suffix}_{window}"
        for window in _WINDOWS
        for suffix in ("relative_spread", "log_event_intensity")
    )
    names = (
        *(f"triplet_imbalance_{name}" for name in triplets),
        *(f"urgency_relative_spread_{name}" for name in urgency),
        *(f"depth_pressure_{name}" for name in depth),
        *flow,
    )
    assert len(names) == M04_FEATURE_COUNT
    return names


def _columns(frame: _FeatureFrame) -> dict[str, np.ndarray]:
    return {
        name: np.asarray(frame.values[:, index], dtype=np.float64)
        for index, name in enumerate(frame.feature_names)
    }


def _ratio_imbalance(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    return (left - right) / (np.abs(left) + np.abs(right) + EPSILON)


def _triplet_imbalance(first: np.ndarray, second: np.ndarray, third: np.ndarray) -> np.ndarray:
    values = np.column_stack((first, second, third))
    high = values.max(axis=1)
    low = values.min(axis=1)
    middle = values.sum(axis=1) - high - low
    return (high - middle) / (middle - low + EPSILON)


def build_optiver_interactions(
    geometry: _FeatureFrame,
    event_flow: _FeatureFrame,
) -> OptiverInteractionFrame:
    """Build exactly 24 finite interactions without realigning either input."""

    geometry.validate()
    event_flow.validate()
    aligned = (
        np.array_equal(geometry.sample_id, event_flow.sample_id)
        and np.array_equal(geometry.month, event_flow.month)
        and np.array_equal(geometry.target, event_flow.target)
    )
    if not aligned:
        raise ValueError("M02-base and M01-A artifacts must be exactly aligned")
    geometry_columns = _columns(geometry)
    event_flow_columns = _columns(event_flow)
    bid1, bid2 = geometry_columns["lob_bid1_depth_share"], geometry_columns["lob_bid2_depth_share"]
    ask1, ask2 = geometry_columns["lob_ask1_depth_share"], geometry_columns["lob_ask2_depth_share"]
    bid_gap = geometry_columns["lob_bid_l1_l2_gap"]
    ask_gap = geometry_columns["lob_ask_l1_l2_gap"]
    volume_ratio = geometry_columns["lob_l1_l2_volume_ratio"]
    l1_imbalance = _ratio_imbalance(bid1, ask1)
    l2_imbalance = _ratio_imbalance(bid2, ask2)
    book_imbalance = _ratio_imbalance(
        geometry_columns["lob_bid_depth_share"], geometry_columns["lob_ask_depth_share"]
    )
    relative_spread_proxy = np.maximum(bid_gap + ask_gap, 0.0)

    order_pressure: dict[int, np.ndarray] = {}
    trade_pressure: dict[int, np.ndarray] = {}
    event_intensity: dict[int, np.ndarray] = {}
    for window in _WINDOWS:
        order_pressure[window] = np.arcsinh(
            event_flow_columns[f"order_signed_volume_per_event_{window}"]
        )
        trade_pressure[window] = np.arcsinh(
            event_flow_columns[f"trade_signed_volume_per_event_{window}"]
        )
        event_intensity[window] = np.maximum(
            event_flow_columns[f"order_event_count_per_second_{window}"]
            + event_flow_columns[f"trade_event_count_per_second_{window}"],
            0.0,
        )
    agreement_5 = np.tanh(order_pressure[5]) * np.tanh(trade_pressure[5])
    columns = [
        _triplet_imbalance(bid1, bid2, ask1),
        _triplet_imbalance(bid1, bid2, ask2),
        _triplet_imbalance(ask1, ask2, bid1),
        _triplet_imbalance(ask1, ask2, bid2),
        _triplet_imbalance(l1_imbalance, l2_imbalance, book_imbalance),
        _triplet_imbalance(bid_gap, ask_gap, volume_ratio),
        relative_spread_proxy * l1_imbalance,
        relative_spread_proxy * l2_imbalance,
        relative_spread_proxy * book_imbalance,
        relative_spread_proxy * order_pressure[5],
        relative_spread_proxy * trade_pressure[5],
        relative_spread_proxy * agreement_5,
        l1_imbalance * bid_gap,
        l2_imbalance * ask_gap,
        book_imbalance * (bid_gap + ask_gap) / 2.0,
        geometry_columns["lob_shape_asymmetry"] * book_imbalance,
    ]
    for window in _WINDOWS:
        combined = order_pressure[window] + trade_pressure[window]
        columns.extend((
            combined * relative_spread_proxy,
            combined * np.log1p(event_intensity[window]),
        ))
    values = np.column_stack(columns).astype(np.float32)
    frame = OptiverInteractionFrame(
        sample_id=np.asarray(geometry.sample_id).copy(),
        month=np.asarray(geometry.month).copy(),
        target=np.asarray(geometry.target, dtype=np.float64).copy(),
        values=values,
        feature_names=optiver_interaction_feature_names(),
    )
    frame.validate()
    return frame


def build_optiver_interactions_file(m02_path: str | Path, m01a_path: str | Path,
                                    labels_path: str | Path | None, output_path: str | Path) -> dict[str, object]:
    """Build the fixed 24-column artifact from already-frozen feature files."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    from ..models.m02 import load_geometry_frame
    from ..models.m01a import load_event_flow_frame

    geometry = load_geometry_frame(m02_path)
    event_flow = load_event_flow_frame(m01a_path)
    frame = build_optiver_interactions(geometry, event_flow)
    output = Path(output_path); output.parent.mkdir(parents=True, exist_ok=True)
    names = optiver_interaction_feature_names()
    pq.write_table(pa.table({"sample_id": frame.sample_id, "month": frame.month, "target": frame.target,
                             **{name: frame.values[:, i] for i, name in enumerate(names)}}), output, compression="zstd")
    diagnostics = {"rows": int(frame.sample_id.size), "feature_names": list(names),
                   "feature_hash": feature_hash(list(names)),
                   "artifact_hashes": {"sample_id": array_hash(frame.sample_id), "month": array_hash(frame.month),
                                       "target": array_hash(frame.target), "values": array_hash(frame.values)},
                   "builder": {"feature_count": 24, "epsilon": EPSILON, "input_alignment": "exact sample_id/month/target"}}
    config_hash = hashlib.sha256(json.dumps(diagnostics, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    ExperimentManifest(experiment_id="m04-optiver-interaction-features", status="complete", config_hash=config_hash,
                       feature_hash=diagnostics["feature_hash"], train_months=(int(frame.month.min()), int(frame.month.max())),
                       diagnostics=diagnostics).write(output.parent)
    (output.parent / "report.md").write_text("# M04 Optiver Interaction Family\n\n- features: `24`\n- inputs: frozen M02-base + M01-A Event Flow\n", encoding="utf-8")
    return diagnostics | {"output": str(output)}


def load_optiver_interaction_frame(path: str | Path) -> OptiverInteractionFrame:
    import pyarrow.parquet as pq
    path = Path(path); manifest = json.loads((path.parent / "manifest.json").read_text(encoding="utf-8"))
    names = optiver_interaction_feature_names()
    if manifest.get("experiment_id") != "m04-optiver-interaction-features" or manifest.get("status") != "complete":
        raise ValueError("M04 feature manifest identity/status is invalid")
    table = pq.read_table(path, columns=["sample_id", "month", "target", *names])
    cols = {name: table[name].to_numpy(zero_copy_only=False) for name in table.column_names}
    order = np.argsort(cols["sample_id"], kind="mergesort")
    cols = {name: np.asarray(value)[order] for name, value in cols.items()}
    frame = OptiverInteractionFrame(cols["sample_id"], cols["month"], cols["target"].astype(np.float64),
                                    np.column_stack([cols[name] for name in names]).astype(np.float32), names)
    frame.validate()
    expected = {"sample_id": array_hash(frame.sample_id), "month": array_hash(frame.month),
                "target": array_hash(frame.target), "values": array_hash(frame.values)}
    if manifest.get("feature_hash") != feature_hash(list(names)) or manifest.get("diagnostics", {}).get("artifact_hashes") != expected:
        raise ValueError("M04 feature manifest hashes are invalid")
    return frame
