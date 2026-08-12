"""Market-centered L1/L2 order-book geometry."""

from __future__ import annotations

from typing import Mapping

import numpy as np


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


def build_lob_geometry(market: Mapping[str, object]) -> tuple[np.ndarray, list[str], np.ndarray]:
    sample_id = np.asarray(market["sample_id"], dtype=np.int64).reshape(-1)
    groups: list[dict[str, float]] = []
    ids = np.array(sorted(np.unique(sample_id)), dtype=np.int64)
    for sid in ids:
        mask = sample_id == sid
        # Use the most recent quote (smallest seconds_before_predict), while
        # preserving a deterministic fallback for equal timestamps.
        indices = np.flatnonzero(mask)
        if "seconds_before_predict" in market:
            seconds = np.asarray(market["seconds_before_predict"], dtype=float).reshape(-1)
            idx = indices[np.argmin(seconds[indices])]
        else:
            idx = indices[-1]
        row = {name: float(np.asarray(values).reshape(-1)[idx]) for name, values in market.items()}
        groups.append(
            lob_geometry_row(
                row["bid_price_1"], row["bid_volume_1"], row["ask_price_1"], row["ask_volume_1"],
                row["bid_price_2"], row["bid_volume_2"], row["ask_price_2"], row["ask_volume_2"],
            )
        )
    names = sorted({name for row in groups for name in row})
    values = np.array([[row[name] for name in names] for row in groups], dtype=np.float32)
    return ids, names, values
