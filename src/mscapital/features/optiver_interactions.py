"""Fixed Optiver-style interactions over frozen M02-base and M01-A inputs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np


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
