"""Dynamic Multi-Level Invariant OFI feature construction.

The public functions work on small numpy arrays as well as arrays extracted
from Arrow tables.  That keeps the definitions unit-testable without scanning
the 100M-row raw files in every test.
"""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np


WINDOWS = (5.0, 15.0, 30.0, 60.0)


def _array(value: object, dtype=float) -> np.ndarray:
    return np.asarray(value, dtype=dtype).reshape(-1)


def signed_order_flow(
    volume: object,
    side: object,
    order_action: object,
) -> np.ndarray:
    """Return signed order flow with bid-add positive convention."""

    v = _array(volume, float)
    s = _array(side, int)
    a = _array(order_action, int)
    if not (v.shape == s.shape == a.shape):
        raise ValueError("volume, side and order_action must have equal lengths")
    side_sign = np.where(s == 0, 1.0, -1.0)
    action_sign = np.where(a == 0, 1.0, -1.0)
    return v * side_sign * action_sign


def signed_trade_flow(volume: object, side: object) -> np.ndarray:
    v = _array(volume, float)
    s = _array(side, int)
    if v.shape != s.shape:
        raise ValueError("volume and side must have equal lengths")
    return v * np.where(s == 0, 1.0, -1.0)


def quote_ofi(
    bid_price: object,
    bid_volume: object,
    ask_price: object,
    ask_volume: object,
) -> np.ndarray:
    """Cont-style quote OFI for one book level.

    Inputs must be in chronological order.  The first observation has zero
    OFI because no previous quote exists.  For a flat bid/ask price the queue
    change contributes with the appropriate bid/ask sign.
    """

    bp, bv, ap, av = (_array(v, float) for v in (bid_price, bid_volume, ask_price, ask_volume))
    if not (bp.shape == bv.shape == ap.shape == av.shape):
        raise ValueError("quote arrays must have equal lengths")
    if bp.size == 0:
        return np.empty(0, dtype=np.float64)
    prev_bp, prev_bv = bp[:-1], bv[:-1]
    prev_ap, prev_av = ap[:-1], av[:-1]
    bid = np.where(bp[1:] > prev_bp, bv[1:], np.where(bp[1:] < prev_bp, -prev_bv, bv[1:] - prev_bv))
    ask = np.where(ap[1:] < prev_ap, av[1:], np.where(ap[1:] > prev_ap, -prev_av, -(av[1:] - prev_av)))
    return np.concatenate(([0.0], bid + ask))


def _window_mask(seconds_before_predict: np.ndarray, window: float) -> np.ndarray:
    # The raw files count backwards from the prediction instant.  The
    # direction is irrelevant for an inclusive lookback window.
    return np.isfinite(seconds_before_predict) & (seconds_before_predict >= 0.0) & (seconds_before_predict <= window)


def _safe_ratio(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator > 0 else 0.0


def _sample_features(
    *,
    order_seconds: np.ndarray,
    order_flow: np.ndarray,
    order_volume: np.ndarray,
    trade_seconds: np.ndarray,
    trade_flow: np.ndarray,
    trade_volume: np.ndarray,
    market_seconds: np.ndarray,
    quote_l1: np.ndarray,
    quote_l2: np.ndarray,
    market_depth: np.ndarray,
) -> dict[str, float]:
    out: dict[str, float] = {}
    order_rates: dict[float, float] = {}
    trade_rates: dict[float, float] = {}
    quote_rates: dict[int, dict[float, float]] = {1: {}, 2: {}}
    for window in WINDOWS:
        om = _window_mask(order_seconds, window)
        tm = _window_mask(trade_seconds, window)
        mm = _window_mask(market_seconds, window)
        order_rate = _safe_ratio(float(order_flow[om].sum()), window)
        trade_rate = _safe_ratio(float(trade_flow[tm].sum()), window)
        order_rates[window] = order_rate
        trade_rates[window] = trade_rate
        out[f"ofi_event_rate_{int(window)}"] = order_rate
        out[f"trade_flow_rate_{int(window)}"] = trade_rate
        out[f"ofi_event_count_rate_{int(window)}"] = _safe_ratio(float(om.sum()), window)
        out[f"trade_event_count_rate_{int(window)}"] = _safe_ratio(float(tm.sum()), window)
        depth = float(market_depth[mm].mean()) if mm.any() else 0.0
        out[f"ofi_depth_{int(window)}"] = _safe_ratio(float(order_flow[om].sum()), abs(depth))
        out[f"ofi_volume_{int(window)}"] = _safe_ratio(float(order_flow[om].sum()), float(order_volume[om].sum()))
        out[f"ofi_trade_volume_{int(window)}"] = _safe_ratio(float(order_flow[om].sum()), float(trade_volume[tm].sum()))
        for level, quote in ((1, quote_l1), (2, quote_l2)):
            qr = _safe_ratio(float(quote[mm].sum()), window)
            quote_rates[level][window] = qr
            out[f"quote_ofi_l{level}_rate_{int(window)}"] = qr
            out[f"quote_ofi_l{level}_depth_{int(window)}"] = _safe_ratio(float(quote[mm].sum()), abs(depth))
    for name, rates in (("ofi_event_rate", order_rates), ("trade_flow_rate", trade_rates)):
        for fast, slow in ((5.0, 15.0), (15.0, 30.0), (30.0, 60.0)):
            out[f"{name}_fast_slow_{int(fast)}_{int(slow)}"] = rates[fast] - rates[slow]
        out[f"{name}_acceleration"] = (rates[5.0] - rates[15.0]) - (rates[15.0] - rates[30.0])
    for level, rates in quote_rates.items():
        for fast, slow in ((5.0, 15.0), (15.0, 30.0), (30.0, 60.0)):
            out[f"quote_ofi_l{level}_fast_slow_{int(fast)}_{int(slow)}"] = rates[fast] - rates[slow]
    out["order_trade_gap_15"] = order_rates[15.0] - trade_rates[15.0]
    out["order_trade_agreement_15"] = order_rates[15.0] * trade_rates[15.0]
    out["quote_trade_agreement_l1_15"] = quote_rates[1][15.0] * trade_rates[15.0]
    out["quote_trade_agreement_l2_15"] = quote_rates[2][15.0] * trade_rates[15.0]
    return out


def _grouped(table: Mapping[str, object], key: str) -> dict[int, dict[str, np.ndarray]]:
    sid = _array(table[key], int)
    result: dict[int, dict[str, np.ndarray]] = {}
    for value in np.unique(sid):
        mask = sid == value
        result[int(value)] = {name: _array(column)[mask] for name, column in table.items() if name != key}
    return result


def build_m01_features(
    order: Mapping[str, object],
    transaction: Mapping[str, object],
    market: Mapping[str, object],
) -> tuple[np.ndarray, list[str], np.ndarray]:
    """Build M01 features for joined order/transaction/market mappings.

    All input mappings must contain ``sample_id`` and the columns needed by
    their source.  Missing order/transaction/market rows produce zeros, which
    is an explicit and stable no-events representation.
    """

    orders, trades, markets = _grouped(order, "sample_id"), _grouped(transaction, "sample_id"), _grouped(market, "sample_id")
    ids = np.array(sorted(set(orders) | set(trades) | set(markets)), dtype=np.int64)
    rows: list[dict[str, float]] = []
    for sample_id in ids:
        o = orders.get(int(sample_id), {})
        t = trades.get(int(sample_id), {})
        m = markets.get(int(sample_id), {})
        of = signed_order_flow(o.get("volume", []), o.get("side", []), o.get("order_action", []))
        tf = signed_trade_flow(t.get("volume", []), t.get("side", []))
        ms = _array(m.get("seconds_before_predict", []), float)
        order_seconds = _array(o.get("seconds_before_predict", []), float)
        trade_seconds = _array(t.get("seconds_before_predict", []), float)
        # Quote OFI requires chronological order; raw files are often stored
        # seconds-before is measured backwards from prediction: larger values
        # are older, so chronological order is descending.
        order_volume = _array(o.get("volume", []), float)
        trade_volume = _array(t.get("volume", []), float)
        market_order = np.argsort(ms)[::-1]
        ms = ms[market_order]
        bid1 = _array(m.get("bid_price_1", []), float)[market_order]
        bv1 = _array(m.get("bid_volume_1", []), float)[market_order]
        ask1 = _array(m.get("ask_price_1", []), float)[market_order]
        av1 = _array(m.get("ask_volume_1", []), float)[market_order]
        bid2 = _array(m.get("bid_price_2", []), float)[market_order]
        bv2 = _array(m.get("bid_volume_2", []), float)[market_order]
        ask2 = _array(m.get("ask_price_2", []), float)[market_order]
        av2 = _array(m.get("ask_volume_2", []), float)[market_order]
        q1 = quote_ofi(bid1, bv1, ask1, av1)
        q2 = quote_ofi(bid2, bv2, ask2, av2)
        depth = bv1 + av1 + bv2 + av2
        rows.append(
            _sample_features(
                order_seconds=order_seconds,
                order_flow=of,
                order_volume=order_volume,
                trade_seconds=trade_seconds,
                trade_flow=tf,
                trade_volume=trade_volume,
                market_seconds=ms,
                quote_l1=q1,
                quote_l2=q2,
                market_depth=depth,
            )
        )
    names = sorted({name for row in rows for name in row})
    values = np.array([[row.get(name, 0.0) for name in names] for row in rows], dtype=np.float32)
    return ids, names, values


def select_m01_stage(
    names: Sequence[str], values: np.ndarray, stage: str
) -> tuple[list[str], np.ndarray]:
    """Select the cumulative M01 ablation stage A-F."""

    stage = stage.upper()
    if stage not in {"A", "B", "C", "D", "E", "F"}:
        raise ValueError("M01 stage must be one of A, B, C, D, E or F")
    raw_windows = {"5", "15", "30", "60"}
    keep: list[str] = []
    for name in names:
        def has_window(prefix: str) -> bool:
            return name.startswith(prefix) and name[len(prefix):] in raw_windows

        is_event = any(has_window(prefix) for prefix in (
            "ofi_event_rate_", "trade_flow_rate_", "ofi_event_count_rate_", "trade_event_count_rate_"
        ))
        is_quote_l1 = has_window("quote_ofi_l1_rate_")
        is_quote_l2 = has_window("quote_ofi_l2_rate_")
        is_invariant = (
            name.startswith(("ofi_depth_", "ofi_volume_", "ofi_trade_volume_", "quote_ofi_l1_depth_", "quote_ofi_l2_depth_"))
            and name.rsplit("_", 1)[-1] in raw_windows
        )
        is_dynamic = any(token in name for token in ("fast_slow", "acceleration"))
        is_cross = name.startswith(("order_trade_", "quote_trade_"))
        allowed = (
            is_event
            or (stage in {"B", "C", "D", "E", "F"} and is_quote_l1)
            or (stage in {"C", "D", "E", "F"} and is_quote_l2)
            or (stage in {"D", "E", "F"} and is_invariant)
            or (stage in {"E", "F"} and is_dynamic)
            or (stage == "F" and is_cross)
        )
        if allowed:
            keep.append(name)
    indices = [names.index(name) for name in keep]
    return keep, np.asarray(values)[:, indices]
