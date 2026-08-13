"""Leakage-safe ReVol-lite market, order and transaction features.

The builder intentionally produces a fixed-width, sample-level representation.
It does not fit any statistics on labels; all normalisation is local to the
look-back window of an individual sample.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np

from ..artifacts import ExperimentManifest, array_hash, feature_hash


WINDOWS: tuple[int, ...] = (5, 15, 30, 60)
WINDOW_FEATURES: tuple[str, ...] = (
    "mid_net_z",
    "mid_range_z",
    "spread_mean_z",
    "microgap_last_z",
    "depth_change_z",
    "order_flow_z",
    "trade_flow_z",
)
CONTEXT_FEATURES: tuple[str, ...] = (
    "log_ret_sigma_60",
    "log_depth_rms_60",
    "log_order_volume_rms_60",
    "log_trade_volume_rms_60",
    "log1p_order_event_rate_60",
    "log1p_trade_event_rate_60",
)
CONTEXT_STATE_FEATURES: tuple[str, ...] = (
    "mid_net_z_60",
    "spread_mean_z_60",
    "depth_change_z_60",
    "order_flow_z_60",
    "trade_flow_z_60",
)
MISSING_FEATURES: tuple[str, ...] = (
    "market_missing",
    "order_missing",
    "trade_missing",
)


def revol_lite_feature_names() -> tuple[str, ...]:
    return tuple(
        [f"revol_{name}_{window}" for window in WINDOWS for name in WINDOW_FEATURES]
        + [f"revol_{name}" for name in CONTEXT_FEATURES]
        + [f"revol_{name}" for name in MISSING_FEATURES]
    )


def context_feature_names() -> tuple[str, ...]:
    return tuple([f"revol_{name}" for name in CONTEXT_FEATURES] + [f"revol_{name}" for name in CONTEXT_STATE_FEATURES])


def _safe_max(value: float, floor: float = 1e-8) -> float:
    return max(float(value) if np.isfinite(value) else 0.0, floor)


def _aggregate_exprs(source: str, windows: tuple[int, ...]) -> list[Any]:
    import polars as pl

    sec = pl.col("seconds_before_predict")
    volume = pl.col("volume").cast(pl.Float64)
    if source == "order":
        signed = volume * pl.when(pl.col("side") == 0).then(1.0).otherwise(-1.0)
        signed = signed * pl.when(pl.col("order_action") == 0).then(1.0).otherwise(-1.0)
    else:
        signed = volume * pl.when(pl.col("side") == 0).then(1.0).otherwise(-1.0)
    output: list[Any] = []
    for window in windows:
        valid = sec.is_finite() & (sec >= 0.0) & (sec <= float(window))
        suffix = str(window)
        output.extend(
            [
                valid.cast(pl.UInt32).sum().alias(f"{source}_count_{suffix}"),
                signed.filter(valid).sum().alias(f"{source}_flow_{suffix}"),
                (volume.pow(2).filter(valid).mean()).alias(f"{source}_volume_sq_mean_{suffix}"),
            ]
        )
    return output


def _aggregate_market(path: str | Path) -> Any:
    import polars as pl

    columns = [
        "sample_id", "seconds_before_predict", "bid_price_1", "bid_volume_1",
        "ask_price_1", "ask_volume_1", "bid_price_2", "bid_volume_2",
        "ask_price_2", "ask_volume_2",
    ]
    frame = pl.scan_ipc(path).select(columns)
    mid = (pl.col("bid_price_1") + pl.col("ask_price_1")) / 2.0
    spread_ratio = (pl.col("ask_price_1") - pl.col("bid_price_1")) / mid
    depth = (
        pl.col("bid_volume_1").cast(pl.Float64)
        + pl.col("ask_volume_1").cast(pl.Float64)
        + pl.col("bid_volume_2").cast(pl.Float64)
        + pl.col("ask_volume_2").cast(pl.Float64)
    )
    valid = (
        mid.is_finite() & (mid > 0.0) & spread_ratio.is_finite() & (spread_ratio >= 0.0)
        & pl.col("bid_price_2").is_finite() & pl.col("ask_price_2").is_finite()
        & pl.col("bid_volume_1").is_finite() & pl.col("ask_volume_1").is_finite()
        & pl.col("bid_volume_2").is_finite() & pl.col("ask_volume_2").is_finite()
        & (pl.col("bid_volume_1") >= 0.0) & (pl.col("ask_volume_1") >= 0.0)
        & (pl.col("bid_volume_2") >= 0.0) & (pl.col("ask_volume_2") >= 0.0)
        & (pl.col("bid_price_2") <= pl.col("bid_price_1"))
        & (pl.col("bid_price_1") < pl.col("ask_price_1"))
        & (pl.col("ask_price_1") <= pl.col("ask_price_2"))
    )
    micro = (
        (pl.col("ask_price_1") * pl.col("bid_volume_1")
         + pl.col("bid_price_1") * pl.col("ask_volume_1"))
        / (pl.col("bid_volume_1") + pl.col("ask_volume_1")).cast(pl.Float64)
        - mid
    ) / mid
    enriched = frame.with_columns(
        mid.alias("_mid"),
        pl.when(valid).then(mid.log()).otherwise(None).alias("_log_mid"),
        pl.when(valid).then(spread_ratio).otherwise(None).alias("_spread_ratio"),
        pl.when(valid).then(depth).otherwise(None).alias("_depth"),
        pl.when(valid).then(micro).otherwise(None).alias("_microgap"),
    ).with_columns(pl.col("_log_mid").diff().over("sample_id").alias("_log_mid_diff"))
    expressions: list[Any] = []
    sec = pl.col("seconds_before_predict")
    for window in WINDOWS:
        valid_window = valid & sec.is_finite() & (sec >= 0.0) & (sec <= float(window))
        suffix = str(window)
        # Source files are sorted oldest-to-newest within each sample.  The
        # first/last aggregate therefore remains causal and avoids a giant sort.
        expressions.extend(
            [
                pl.col("_log_mid").filter(valid_window).first().alias(f"market_log_mid_first_{suffix}"),
                pl.col("_log_mid").filter(valid_window).last().alias(f"market_log_mid_last_{suffix}"),
                pl.col("_log_mid").filter(valid_window).min().alias(f"market_log_mid_min_{suffix}"),
                pl.col("_log_mid").filter(valid_window).max().alias(f"market_log_mid_max_{suffix}"),
                pl.col("_log_mid_diff").filter(valid_window).pow(2).mean().sqrt().alias(f"market_log_diff_rms_{suffix}"),
                pl.col("_spread_ratio").filter(valid_window).median().alias(f"market_spread_median_{suffix}"),
                pl.col("_spread_ratio").filter(valid_window).mean().alias(f"market_spread_mean_{suffix}"),
                pl.col("_microgap").filter(valid_window).last().alias(f"market_micro_last_{suffix}"),
                pl.col("_depth").filter(valid_window).first().alias(f"market_depth_first_{suffix}"),
                pl.col("_depth").filter(valid_window).last().alias(f"market_depth_last_{suffix}"),
                pl.col("_depth").filter(valid_window).pow(2).mean().sqrt().alias(f"market_depth_rms_{suffix}"),
                valid_window.cast(pl.UInt32).sum().alias(f"market_count_{suffix}"),
            ]
        )
    return enriched.group_by("sample_id", maintain_order=True).agg(expressions).collect(engine="streaming")


def _aggregate_events(path: str | Path, source: str) -> Any:
    import polars as pl

    required = ["sample_id", "seconds_before_predict", "volume", "side"]
    if source == "order":
        required.append("order_action")
    frame = pl.scan_ipc(path).select(required)
    invalid = ~pl.col("side").is_in([0, 1])
    if source == "order":
        invalid = invalid | ~pl.col("order_action").is_in([0, 1])
    aggregated = frame.group_by("sample_id", maintain_order=True).agg(
        _aggregate_exprs(source, WINDOWS) + [invalid.cast(pl.UInt32).sum().alias("_invalid_code_count")]
    ).collect(engine="streaming")
    if int(aggregated["_invalid_code_count"].sum()) != 0:
        raise ValueError(f"{source} side/action contains a code outside {{0, 1}}")
    return aggregated.drop("_invalid_code_count")


def _column(table: Any, name: str, rows: int) -> np.ndarray:
    if name not in table.columns:
        return np.full(rows, np.nan, dtype=np.float64)
    return table[name].to_numpy().astype(np.float64, copy=False)


def _read_ipc_columns(path: str | Path, columns: list[str]) -> dict[str, np.ndarray]:
    """Small PyArrow fallback for environments without the optional Polars wheel."""
    import pyarrow.feather as feather

    table = feather.read_table(path, columns=columns, memory_map=True)
    return {name: table[name].to_numpy(zero_copy_only=False) for name in columns}


def _group_slices(sample_id: np.ndarray) -> tuple[np.ndarray, list[slice]]:
    order = np.argsort(sample_id, kind="stable")
    sorted_ids = np.asarray(sample_id)[order]
    unique, starts = np.unique(sorted_ids, return_index=True)
    ends = np.r_[starts[1:], sorted_ids.size]
    return unique, [slice(int(start), int(end)) for start, end in zip(starts, ends)]


def _aggregate_market_numpy(path: str | Path) -> dict[str, np.ndarray]:
    columns = [
        "sample_id", "seconds_before_predict", "bid_price_1", "bid_volume_1",
        "ask_price_1", "ask_volume_1", "bid_price_2", "bid_volume_2",
        "ask_price_2", "ask_volume_2",
    ]
    raw = _read_ipc_columns(path, columns)
    ids, slices = _group_slices(raw["sample_id"])
    order = np.argsort(raw["sample_id"], kind="stable")
    raw = {name: np.asarray(value)[order] for name, value in raw.items()}
    result: dict[str, np.ndarray] = {"sample_id": ids.astype(np.int64, copy=False)}
    for window in WINDOWS:
        suffix = str(window)
        for name in (
            "log_mid_first", "log_mid_last", "log_mid_min", "log_mid_max",
            "log_diff_rms", "spread_median", "spread_mean", "micro_last",
            "depth_first", "depth_last", "depth_rms",
        ):
            result[f"market_{name}_{suffix}"] = np.full(ids.size, np.nan, dtype=np.float64)
        result[f"market_count_{suffix}"] = np.zeros(ids.size, dtype=np.int64)
    for row, span in enumerate(slices):
        sec = np.asarray(raw["seconds_before_predict"][span], dtype=float)
        bid1 = np.asarray(raw["bid_price_1"][span], dtype=float)
        ask1 = np.asarray(raw["ask_price_1"][span], dtype=float)
        bid2 = np.asarray(raw["bid_price_2"][span], dtype=float)
        ask2 = np.asarray(raw["ask_price_2"][span], dtype=float)
        bidv1 = np.asarray(raw["bid_volume_1"][span], dtype=float)
        askv1 = np.asarray(raw["ask_volume_1"][span], dtype=float)
        bidv2 = np.asarray(raw["bid_volume_2"][span], dtype=float)
        askv2 = np.asarray(raw["ask_volume_2"][span], dtype=float)
        mid = (bid1 + ask1) / 2.0
        spread = (ask1 - bid1) / mid
        valid = (
            np.isfinite(mid) & (mid > 0.0) & np.isfinite(spread) & (spread >= 0.0)
            & np.isfinite(bid2) & np.isfinite(ask2)
            & np.isfinite(bidv1) & np.isfinite(askv1) & np.isfinite(bidv2) & np.isfinite(askv2)
            & (bidv1 >= 0.0) & (askv1 >= 0.0) & (bidv2 >= 0.0) & (askv2 >= 0.0)
            & (bid2 <= bid1)
            & (bid1 < ask1) & (ask1 <= ask2)
        )
        logmid = np.full(mid.size, np.nan)
        logmid[valid] = np.log(mid[valid])
        depth = bidv1 + askv1 + bidv2 + askv2
        denom = bidv1 + askv1
        micro = np.full(mid.size, np.nan)
        valid_micro = valid & np.isfinite(denom) & (denom > 0.0)
        micro[valid_micro] = ((ask1[valid_micro] * bidv1[valid_micro] + bid1[valid_micro] * askv1[valid_micro]) / denom[valid_micro] - mid[valid_micro]) / mid[valid_micro]
        logdiff = np.full(mid.size, np.nan)
        valid_log = np.flatnonzero(np.isfinite(logmid))
        if valid_log.size > 1:
            logdiff[valid_log[1:]] = np.diff(logmid[valid_log])
        for window in WINDOWS:
            mask = valid & np.isfinite(sec) & (sec >= 0.0) & (sec <= float(window))
            suffix = str(window)
            indices = np.flatnonzero(mask)
            result[f"market_count_{suffix}"][row] = indices.size
            if indices.size == 0:
                continue
            def stat(values: np.ndarray, fn: Any) -> float:
                selected = np.asarray(values[indices], dtype=float)
                return float(fn(selected)) if selected.size else np.nan
            result[f"market_log_mid_first_{suffix}"][row] = stat(logmid, lambda x: x[0])
            result[f"market_log_mid_last_{suffix}"][row] = stat(logmid, lambda x: x[-1])
            result[f"market_log_mid_min_{suffix}"][row] = stat(logmid, np.min)
            result[f"market_log_mid_max_{suffix}"][row] = stat(logmid, np.max)
            result[f"market_log_diff_rms_{suffix}"][row] = stat(logdiff, lambda x: np.sqrt(np.mean(x[np.isfinite(x)] ** 2)) if np.isfinite(x).any() else np.nan)
            result[f"market_spread_median_{suffix}"][row] = stat(spread, np.median)
            result[f"market_spread_mean_{suffix}"][row] = stat(spread, np.mean)
            result[f"market_micro_last_{suffix}"][row] = stat(micro, lambda x: x[-1])
            result[f"market_depth_first_{suffix}"][row] = stat(depth, lambda x: x[0])
            result[f"market_depth_last_{suffix}"][row] = stat(depth, lambda x: x[-1])
            result[f"market_depth_rms_{suffix}"][row] = stat(depth, lambda x: np.sqrt(np.mean(x ** 2)))
    return result


def _aggregate_events_numpy(path: str | Path, source: str) -> dict[str, np.ndarray]:
    required = ["sample_id", "seconds_before_predict", "volume", "side"]
    if source == "order":
        required.append("order_action")
    raw = _read_ipc_columns(path, required)
    if not np.isin(raw["side"], [0, 1]).all():
        raise ValueError(f"{source} side contains a code outside {{0, 1}}")
    if source == "order" and not np.isin(raw["order_action"], [0, 1]).all():
        raise ValueError("order action contains a code outside {0, 1}")
    ids, slices = _group_slices(raw["sample_id"])
    order = np.argsort(raw["sample_id"], kind="stable")
    raw = {name: np.asarray(value)[order] for name, value in raw.items()}
    result: dict[str, np.ndarray] = {"sample_id": ids.astype(np.int64, copy=False)}
    for window in WINDOWS:
        suffix = str(window)
        result[f"{source}_count_{suffix}"] = np.zeros(ids.size, dtype=np.int64)
        result[f"{source}_flow_{suffix}"] = np.zeros(ids.size, dtype=np.float64)
        result[f"{source}_volume_sq_mean_{suffix}"] = np.full(ids.size, np.nan, dtype=np.float64)
    for row, span in enumerate(slices):
        sec = np.asarray(raw["seconds_before_predict"][span], dtype=float)
        volume = np.asarray(raw["volume"][span], dtype=float)
        signed = volume * np.where(np.asarray(raw["side"][span]) == 0, 1.0, -1.0)
        if source == "order":
            signed *= np.where(np.asarray(raw["order_action"][span]) == 0, 1.0, -1.0)
        for window in WINDOWS:
            mask = np.isfinite(sec) & (sec >= 0.0) & (sec <= float(window))
            suffix = str(window)
            result[f"{source}_count_{suffix}"][row] = int(mask.sum())
            result[f"{source}_flow_{suffix}"][row] = float(np.nansum(signed[mask]))
            if mask.any():
                result[f"{source}_volume_sq_mean_{suffix}"][row] = float(np.nanmean(volume[mask] ** 2))
    return result


def _build_values_numpy(market: dict[str, np.ndarray], order: dict[str, np.ndarray], trade: dict[str, np.ndarray], labels: dict[str, np.ndarray]) -> tuple[np.ndarray, tuple[str, ...], np.ndarray]:
    label_order = np.argsort(labels["sample_id"], kind="stable")
    ids = np.asarray(labels["sample_id"])[label_order].astype(np.int64, copy=False)
    n = ids.size
    names = revol_lite_feature_names()
    values = np.zeros((n, len(names)), dtype=np.float32)
    name_to_index = {name: index for index, name in enumerate(names)}

    def aligned(table: dict[str, np.ndarray], name: str) -> np.ndarray:
        table_ids = np.asarray(table["sample_id"])
        if table_ids.size == 0:
            return np.full(n, np.nan, dtype=np.float64)
        positions = np.searchsorted(table_ids, ids)
        present = positions < table_ids.size
        present &= np.where(positions < table_ids.size, table_ids[np.minimum(positions, table_ids.size - 1)] == ids, False)
        result = np.full(n, np.nan, dtype=np.float64)
        result[present] = np.asarray(table[name], dtype=float)[positions[present]]
        return result

    market_missing = np.ones(n, dtype=np.float64)
    order_missing = np.ones(n, dtype=np.float64)
    trade_missing = np.ones(n, dtype=np.float64)
    for window in WINDOWS:
        s = str(window)
        log_first, log_last = aligned(market, f"market_log_mid_first_{s}"), aligned(market, f"market_log_mid_last_{s}")
        log_min, log_max = aligned(market, f"market_log_mid_min_{s}"), aligned(market, f"market_log_mid_max_{s}")
        price_rms, spread_median = aligned(market, f"market_log_diff_rms_{s}"), aligned(market, f"market_spread_median_{s}")
        spread_mean, micro = aligned(market, f"market_spread_mean_{s}"), aligned(market, f"market_micro_last_{s}")
        depth_first, depth_last = aligned(market, f"market_depth_first_{s}"), aligned(market, f"market_depth_last_{s}")
        depth_rms, market_count = aligned(market, f"market_depth_rms_{s}"), aligned(market, f"market_count_{s}")
        price_scale = np.maximum(np.nan_to_num(price_rms, nan=0.0), np.nan_to_num(spread_median, nan=0.0))
        price_scale = np.maximum(price_scale, 1e-8)
        depth_scale = np.maximum(np.nan_to_num(depth_rms, nan=0.0), 1e-8)
        valid_market = np.isfinite(log_first) & np.isfinite(log_last) & (market_count > 0)
        market_missing[valid_market] = 0.0
        features = {
            f"revol_mid_net_z_{window}": (np.nan_to_num(log_last) - np.nan_to_num(log_first)) / price_scale,
            f"revol_mid_range_z_{window}": (np.nan_to_num(log_max) - np.nan_to_num(log_min)) / price_scale,
            f"revol_spread_mean_z_{window}": np.nan_to_num(spread_mean) / price_scale,
            f"revol_microgap_last_z_{window}": np.nan_to_num(micro) / price_scale,
            f"revol_depth_change_z_{window}": (np.nan_to_num(depth_last) - np.nan_to_num(depth_first)) / depth_scale,
        }
        for name, column in features.items():
            values[:, name_to_index[name]] = np.nan_to_num(column, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        order_count, order_flow, order_vol_sq = aligned(order, f"order_count_{s}"), aligned(order, f"order_flow_{s}"), aligned(order, f"order_volume_sq_mean_{s}")
        trade_count, trade_flow, trade_vol_sq = aligned(trade, f"trade_count_{s}"), aligned(trade, f"trade_flow_{s}"), aligned(trade, f"trade_volume_sq_mean_{s}")
        order_scale = np.maximum(np.sqrt(np.maximum(np.nan_to_num(order_count), 0.0) * np.nan_to_num(order_vol_sq, nan=0.0)), 1e-8)
        trade_scale = np.maximum(np.sqrt(np.maximum(np.nan_to_num(trade_count), 0.0) * np.nan_to_num(trade_vol_sq, nan=0.0)), 1e-8)
        values[:, name_to_index[f"revol_order_flow_z_{window}"]] = np.nan_to_num(order_flow, nan=0.0) / order_scale
        values[:, name_to_index[f"revol_trade_flow_z_{window}"]] = np.nan_to_num(trade_flow, nan=0.0) / trade_scale
        order_missing[order_count > 0] = 0.0
        trade_missing[trade_count > 0] = 0.0
        if window == 60:
            context = {
                "revol_log_ret_sigma_60": np.log(np.maximum(np.nan_to_num(price_rms, nan=0.0), 1e-8)),
                "revol_log_depth_rms_60": np.log(np.maximum(np.nan_to_num(depth_rms, nan=0.0), 1e-8)),
                "revol_log_order_volume_rms_60": np.log(np.maximum(np.sqrt(np.nan_to_num(order_vol_sq, nan=0.0)), 1e-8)),
                "revol_log_trade_volume_rms_60": np.log(np.maximum(np.sqrt(np.nan_to_num(trade_vol_sq, nan=0.0)), 1e-8)),
                "revol_log1p_order_event_rate_60": np.log1p(np.maximum(np.nan_to_num(order_count), 0.0) / 60.0),
                "revol_log1p_trade_event_rate_60": np.log1p(np.maximum(np.nan_to_num(trade_count), 0.0) / 60.0),
            }
            context.update({f"revol_{key}": values[:, name_to_index[f"revol_{key}"]] for key in CONTEXT_STATE_FEATURES})
            for name, column in context.items():
                values[:, name_to_index[name]] = np.nan_to_num(column, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    values[:, name_to_index["revol_market_missing"]] = market_missing.astype(np.float32)
    values[:, name_to_index["revol_order_missing"]] = order_missing.astype(np.float32)
    values[:, name_to_index["revol_trade_missing"]] = trade_missing.astype(np.float32)
    if not np.isfinite(values).all():
        raise ValueError("ReVol-lite features must be finite")
    return ids, names, values


def _build_values(market: Any, order: Any, trade: Any, labels: Any) -> tuple[np.ndarray, tuple[str, ...], np.ndarray]:
    import polars as pl

    joined = (
        labels.select("sample_id", "month", "target")
        .join(market, on="sample_id", how="left")
        .join(order, on="sample_id", how="left")
        .join(trade, on="sample_id", how="left")
        .sort("sample_id")
    )
    ids = joined["sample_id"].to_numpy().astype(np.int64, copy=False)
    n = ids.size
    values = np.zeros((n, len(revol_lite_feature_names())), dtype=np.float32)
    names = revol_lite_feature_names()
    name_to_index = {name: index for index, name in enumerate(names)}
    market_missing = np.ones(n, dtype=np.float64)
    order_missing = np.ones(n, dtype=np.float64)
    trade_missing = np.ones(n, dtype=np.float64)

    for window in WINDOWS:
        s = str(window)
        log_first = _column(joined, f"market_log_mid_first_{s}", n)
        log_last = _column(joined, f"market_log_mid_last_{s}", n)
        log_min = _column(joined, f"market_log_mid_min_{s}", n)
        log_max = _column(joined, f"market_log_mid_max_{s}", n)
        price_rms = _column(joined, f"market_log_diff_rms_{s}", n)
        spread_median = _column(joined, f"market_spread_median_{s}", n)
        spread_mean = _column(joined, f"market_spread_mean_{s}", n)
        micro = _column(joined, f"market_micro_last_{s}", n)
        depth_first = _column(joined, f"market_depth_first_{s}", n)
        depth_last = _column(joined, f"market_depth_last_{s}", n)
        depth_rms = _column(joined, f"market_depth_rms_{s}", n)
        market_count = _column(joined, f"market_count_{s}", n)
        price_scale = np.maximum(np.nan_to_num(price_rms, nan=0.0), np.nan_to_num(spread_median, nan=0.0))
        price_scale = np.maximum(price_scale, 1e-8)
        depth_scale = np.maximum(np.nan_to_num(depth_rms, nan=0.0), 1e-8)
        valid_market = np.isfinite(log_first) & np.isfinite(log_last) & (market_count > 0)
        market_missing[valid_market] = 0.0
        features = {
            f"revol_mid_net_z_{window}": (np.nan_to_num(log_last) - np.nan_to_num(log_first)) / price_scale,
            f"revol_mid_range_z_{window}": (np.nan_to_num(log_max) - np.nan_to_num(log_min)) / price_scale,
            f"revol_spread_mean_z_{window}": np.nan_to_num(spread_mean) / price_scale,
            f"revol_microgap_last_z_{window}": np.nan_to_num(micro) / price_scale,
            f"revol_depth_change_z_{window}": (np.nan_to_num(depth_last) - np.nan_to_num(depth_first)) / depth_scale,
        }
        for name, column in features.items():
            values[:, name_to_index[name]] = np.nan_to_num(column, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

        order_count = _column(joined, f"order_count_{s}", n)
        order_flow = _column(joined, f"order_flow_{s}", n)
        order_vol_sq = _column(joined, f"order_volume_sq_mean_{s}", n)
        trade_count = _column(joined, f"trade_count_{s}", n)
        trade_flow = _column(joined, f"trade_flow_{s}", n)
        trade_vol_sq = _column(joined, f"trade_volume_sq_mean_{s}", n)
        order_scale = np.maximum(np.sqrt(np.maximum(np.nan_to_num(order_count, nan=0.0), 0.0) * np.nan_to_num(order_vol_sq, nan=0.0)), 1e-8)
        trade_scale = np.maximum(np.sqrt(np.maximum(np.nan_to_num(trade_count, nan=0.0), 0.0) * np.nan_to_num(trade_vol_sq, nan=0.0)), 1e-8)
        values[:, name_to_index[f"revol_order_flow_z_{window}"]] = np.nan_to_num(order_flow, nan=0.0) / order_scale
        values[:, name_to_index[f"revol_trade_flow_z_{window}"]] = np.nan_to_num(trade_flow, nan=0.0) / trade_scale
        order_missing[order_count > 0] = 0.0
        trade_missing[trade_count > 0] = 0.0

        if window == 60:
            context = {
                "revol_log_ret_sigma_60": np.log(np.maximum(np.nan_to_num(price_rms, nan=0.0), 1e-8)),
                "revol_log_depth_rms_60": np.log(np.maximum(np.nan_to_num(depth_rms, nan=0.0), 1e-8)),
                "revol_log_order_volume_rms_60": np.log(np.maximum(np.sqrt(np.nan_to_num(order_vol_sq, nan=0.0)), 1e-8)),
                "revol_log_trade_volume_rms_60": np.log(np.maximum(np.sqrt(np.nan_to_num(trade_vol_sq, nan=0.0)), 1e-8)),
                "revol_log1p_order_event_rate_60": np.log1p(np.maximum(order_count, 0.0) / 60.0),
                "revol_log1p_trade_event_rate_60": np.log1p(np.maximum(trade_count, 0.0) / 60.0),
            }
            context.update({f"revol_{key}": values[:, name_to_index[f"revol_{key}"]] for key in CONTEXT_STATE_FEATURES})
            for name, column in context.items():
                values[:, name_to_index[name]] = np.nan_to_num(column, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    values[:, name_to_index["revol_market_missing"]] = market_missing.astype(np.float32)
    values[:, name_to_index["revol_order_missing"]] = order_missing.astype(np.float32)
    values[:, name_to_index["revol_trade_missing"]] = trade_missing.astype(np.float32)
    if not np.isfinite(values).all():
        raise ValueError("ReVol-lite features must be finite")
    return ids, names, values


def build_revol_lite_file(
    market_path: str | Path,
    order_path: str | Path,
    transaction_path: str | Path,
    labels_path: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    """Build the fixed-width ReVol-lite artifact from raw training files."""

    try:
        import polars as pl
    except ImportError:  # pragma: no cover - exercised when the optional wheel is absent
        pl = None
    except RuntimeError as exc:  # pragma: no cover - old CPUs and some wheels misreport flags
        if "feature flag" not in str(exc).lower():
            raise
        os.environ.setdefault("POLARS_SKIP_CPU_CHECK", "1")
        import polars as pl
    import pyarrow as pa
    import pyarrow.parquet as pq

    started = time.perf_counter()
    if pl is not None and hasattr(pl, "read_ipc") and hasattr(pl, "scan_ipc"):
        labels = pl.read_ipc(labels_path).select("sample_id", "month", "target")
        if labels["sample_id"].n_unique() != labels.height:
            raise ValueError("labels sample_id must be unique")
        if not np.isfinite(labels["target"].to_numpy()).all():
            raise ValueError("labels target must be finite")
        market = _aggregate_market(market_path)
        order = _aggregate_events(order_path, "order")
        trade = _aggregate_events(transaction_path, "trade")
        ids, names, values = _build_values(market, order, trade, labels)
        labels_sorted = labels.sort("sample_id")
        month = labels_sorted["month"].to_numpy().astype(np.int16, copy=False)
        target = labels_sorted["target"].to_numpy().astype(np.float32, copy=False)
        label_ids = labels_sorted["sample_id"].to_numpy()
    else:
        label_data = _read_ipc_columns(labels_path, ["sample_id", "month", "target"])
        if np.unique(label_data["sample_id"]).size != label_data["sample_id"].size:
            raise ValueError("labels sample_id must be unique")
        if not np.isfinite(label_data["target"]).all():
            raise ValueError("labels target must be finite")
        market = _aggregate_market_numpy(market_path)
        order = _aggregate_events_numpy(order_path, "order")
        trade = _aggregate_events_numpy(transaction_path, "trade")
        ids, names, values = _build_values_numpy(market, order, trade, label_data)
        label_order = np.argsort(label_data["sample_id"], kind="stable")
        label_ids = np.asarray(label_data["sample_id"])[label_order]
        month = np.asarray(label_data["month"])[label_order].astype(np.int16, copy=False)
        target = np.asarray(label_data["target"])[label_order].astype(np.float32, copy=False)
    if not np.array_equal(ids, label_ids):
        raise ValueError("ReVol-lite artifact does not cover labels exactly")
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(
        pa.table({"sample_id": ids, "month": month, "target": target, **{
            name: values[:, index] for index, name in enumerate(names)
        }}),
        output, compression="zstd", write_statistics=True,
    )
    diagnostics = {
        "rows": int(ids.size), "months": [int(month.min()), int(month.max())],
        "feature_names": names, "feature_hash": feature_hash(names),
        "artifact_hashes": {"sample_id": array_hash(ids), "month": array_hash(month), "target": array_hash(target), "values": array_hash(values)},
        "windows": list(WINDOWS), "feature_count": len(names), "normalization": "window-local ReVol-lite; no target fitting",
    }
    config_hash = hashlib.sha256(json.dumps(diagnostics, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    ExperimentManifest(
        experiment_id="e01-revol-lite-features", status="complete", config_hash=config_hash,
        data_fingerprints={Path(path).name: _file_fingerprint(path) for path in (market_path, order_path, transaction_path, labels_path)},
        feature_hash=diagnostics["feature_hash"], train_months=tuple(diagnostics["months"]), diagnostics=diagnostics,
        runtime_seconds=time.perf_counter() - started,
    ).write(output.parent)
    (output.parent / "report.md").write_text("\n".join([
        "# E01 ReVol-lite features", "", f"- rows: `{ids.size}`", f"- features: `{len(names)}`",
        f"- months: `{diagnostics['months'][0]}-{diagnostics['months'][1]}`", "- status: `complete; target-free feature construction`", "",
    ]), encoding="utf-8")
    return diagnostics | {"output": str(output), "result_hash": config_hash}


def _file_fingerprint(path: str | Path, sample_bytes: int = 1 << 20) -> str:
    target = Path(path)
    stat = target.stat()
    digest = hashlib.sha256()
    digest.update(str(stat.st_size).encode("ascii"))
    with target.open("rb") as handle:
        digest.update(handle.read(sample_bytes))
        if stat.st_size > sample_bytes:
            handle.seek(max(0, stat.st_size - sample_bytes))
            digest.update(handle.read(sample_bytes))
    return digest.hexdigest()
